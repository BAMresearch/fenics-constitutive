"""
Tests for the Central Difference Method (CDM) explicit dynamics solver.

We test a 1D linear elastic bar problem which has an analytical solution.
A bar of length L is fixed at x=0 and subjected to a suddenly applied 
constant displacement at x=L. The wave propagates through the bar and 
the displacement can be compared to the analytical solution.
"""

from __future__ import annotations

from typing import cast

import dolfinx as df
import numpy as np
import pytest
from mpi4py import MPI

from fenics_constitutive.models.interfaces import StressStrainConstraint
from fenics_constitutive.models.linear_elasticity_model import LinearElasticityModel
from fenics_constitutive.solver import IncrSmallStrainProblem
from fenics_constitutive.solver.central_difference_method import CDMSolver


def analytical_free_vibration_solution(
    x: np.ndarray,
    t: float,
    L: float,
    c: float,
    amplitude: float,
    mode: int = 1,
) -> np.ndarray:
    """
    Analytical solution for free vibration of a fixed-fixed 1D bar.
    
    The bar is fixed at both ends with initial displacement in the nth mode shape.
    
    PDE: rho * d^2u/dt^2 = E * d^2u/dx^2
    
    Boundary conditions:
        u(0, t) = 0
        u(L, t) = 0
    
    Initial conditions:
        u(x, 0) = A * sin(n*pi*x/L)  (nth mode shape)
        du/dt(x, 0) = 0
    
    Solution:
        u(x, t) = A * sin(n*pi*x/L) * cos(omega_n * t)
        
    where omega_n = n * pi * c / L is the nth natural frequency.
    
    Args:
        x: Position array
        t: Time
        L: Bar length
        c: Wave speed sqrt(E/rho)
        amplitude: Initial displacement amplitude
        mode: Mode number (1, 2, 3, ...)
    
    Returns:
        Displacement at positions x at time t
    """
    omega_n = mode * np.pi * c / L
    return amplitude * np.sin(mode * np.pi * x / L) * np.cos(omega_n * t)

@pytest.mark.parametrize("n_elements", [20, 40])
def test_cdm_1d_free_vibration(n_elements: int):
    """
    Test CDM solver against analytical solution for free vibration.
    
    A fixed-fixed bar with initial sinusoidal displacement (1st mode)
    vibrates harmonically. This test has no infinite accelerations.
    """
    # Material and geometry parameters
    E = 210e9  # Young's modulus (Pa)
    nu = 0.3   # Poisson's ratio
    rho = 7800.0  # Density (kg/m^3)
    L = 1.0  # Bar length (m)
    amplitude = 0.001  # Initial displacement amplitude (m)
    mode = 1  # First mode
    
    # Wave speed
    c = np.sqrt(E / rho)
    
    # Natural frequency and period
    omega_1 = mode * np.pi * c / L
    T_period = 2 * np.pi / omega_1
    
    # Simulate for one full period
    T_end = T_period
    
    # Create 1D mesh
    mesh = df.mesh.create_interval(MPI.COMM_WORLD, n_elements, [0.0, L])
    
    # Scalar function space
    V = df.fem.functionspace(mesh, ("Lagrange", 1))
    u = cast(df.fem.Function,df.fem.Function(V, name="Displacement"))
    
    # Linear elastic law
    law = LinearElasticityModel(
        parameters={"E": E, "nu": nu},
        constraint=StressStrainConstraint.UNIAXIAL_STRESS,
    )
    
    # Boundary conditions (both ends fixed)
    def left(x):
        return np.isclose(x[0], 0.0)
    
    def right(x):
        return np.isclose(x[0], L)
    
    dofs_left = df.fem.locate_dofs_geometrical(V, left)
    dofs_right = df.fem.locate_dofs_geometrical(V, right)
    
    bc_left = df.fem.dirichletbc(df.fem.Constant(mesh, np.float64(0.0)), dofs_left, V)
    bc_right = df.fem.dirichletbc(df.fem.Constant(mesh, np.float64(0.0)), dofs_right, V)
    bcs = [bc_left, bc_right]
    
    # Create problem
    problem = IncrSmallStrainProblem(law, u, bcs, q_degree=1)
    
    # Set initial displacement (1st mode shape)
    x_coords = V.tabulate_dof_coordinates()[:, 0]
    u_initial = amplitude * np.sin(mode * np.pi * x_coords / L)
    
    # Create initial displacement function
    u0 = cast(df.fem.Function, df.fem.Function(V, name="InitialDisplacement"))
    u0.x.array[:] = u_initial
    # Ensure BCs are satisfied in initial condition
    u0.x.array[dofs_left] = 0.0
    u0.x.array[dofs_right] = 0.0
    u0.x.scatter_forward()
    
    # Create initial velocity function (zero)
    v0 = cast(df.fem.Function, df.fem.Function(V, name="InitialVelocity"))
    v0.x.array[:] = 0.0
    v0.x.scatter_forward()
    
    # Create CDM solver
    safety_factor = 0.8
    solver = CDMSolver(problem, density=rho, u0=u0, v0=v0, safety_factor=safety_factor)
    
    # Time stepping
    dt = problem.sim_time.dt
    n_steps = int(T_end / dt)
    
    for _ in range(n_steps):
        solver.step()
        problem.update()
    
    # Get numerical solution
    u_numerical = problem.incr_disp.current.x.array.copy()
    
    # Compute analytical solution at final time
    t_final = n_steps * dt
    u_analytical = analytical_free_vibration_solution(
        x_coords, t_final, L, c, amplitude, mode
    )
    
    # After one period, displacement should return to initial shape
    # Compare numerical to analytical
    error = np.abs(u_numerical - u_analytical)
    max_error = np.max(error)
    rel_error = max_error / amplitude

    # Allow some numerical dissipation/dispersion error
    assert rel_error < 1e-5, f"Relative error {rel_error:.2e} too large"



if __name__=="__main__":
    test_cdm_1d_free_vibration(20)