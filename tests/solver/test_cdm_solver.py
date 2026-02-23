"""
Tests for the Central Difference Method (CDM) explicit dynamics solver.

We test a 1D linear elastic bar problem which has an analytical solution.
A bar of length L is fixed at x=0 and subjected to a suddenly applied 
constant displacement at x=L. The wave propagates through the bar and 
the displacement can be compared to the analytical solution.
"""

from __future__ import annotations

import dolfinx as df
import numpy as np
import pytest
from mpi4py import MPI

from fenics_constitutive.models.linear_elasticity_model import LinearElasticityModel
from fenics_constitutive.models.interfaces import StressStrainConstraint
from fenics_constitutive.solver import IncrSmallStrainProblem
from fenics_constitutive.solver.central_difference_method import CDMSolver


# def analytical_1d_wave_solution(
#     x: np.ndarray,
#     t: float,
#     L: float,
#     c: float,
#     u_applied: float,
#     n_terms: int = 100,
# ) -> np.ndarray:
#     """
#     Analytical solution for a 1D bar with one end fixed and the other 
#     subjected to a suddenly applied displacement u_applied at t=0.
    
#     The PDE is:
#         rho * d^2u/dt^2 = E * d^2u/dx^2
    
#     with boundary conditions:
#         u(0, t) = 0  (fixed end)
#         u(L, t) = u_applied  (applied displacement for t > 0)
    
#     and initial conditions:
#         u(x, 0) = 0
#         du/dt(x, 0) = 0
    
#     The solution is:
#         u(x, t) = u_applied * x/L - (2*u_applied/pi) * sum_{n=1}^{inf} 
#                   [(-1)^n / n] * sin(n*pi*x/L) * cos(n*pi*c*t/L)
    
#     Args:
#         x: Position array
#         t: Time
#         L: Bar length
#         c: Wave speed (sqrt(E/rho)) for uniaxial stress
#         u_applied: Applied displacement at x=L
#         n_terms: Number of terms in the Fourier series
    
#     Returns:
#         Displacement at positions x at time t
#     """
#     # Static solution (first term)
#     u = u_applied * x / L
    
#     # Dynamic correction (Fourier series)
#     for n in range(1, n_terms + 1):
#         coeff = (2 * u_applied / np.pi) * ((-1) ** n / n)
#         u -= coeff * np.sin(n * np.pi * x / L) * np.cos(n * np.pi * c * t / L)
    
#     return u


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


# @pytest.mark.parametrize("n_elements", [10, 20])
# def test_cdm_1d_linear_elastic_wave(n_elements: int):
#     """
#     Test the CDM solver against analytical solution for 1D wave propagation.
    
#     A 1D bar is fixed at x=0 and a constant displacement is applied at x=L.
#     After some time, we compare the numerical solution to the analytical one.
#     Uses UNIAXIAL_STRESS constraint where wave speed c = sqrt(E/rho).
#     """
#     # Material and geometry parameters
#     E = 210e9  # Young's modulus (Pa)
#     nu = 0.3   # Poisson's ratio (not used for uniaxial stress)
#     rho = 7800.  # Density (kg/m^3)
#     L = 1.0  # Bar length (m)
#     u_applied = 0.001  # Applied displacement (m)
    
#     # Wave speed for uniaxial stress: c = sqrt(E/rho)
#     c = np.sqrt(E / rho)
    
#     # Time parameters - simulate for a short time
#     T_end = 0.5 * L / c  # Time for wave to travel halfway
    
#     # Create 1D mesh
#     mesh = df.mesh.create_interval(MPI.COMM_WORLD, n_elements, [0.0, L])
    
#     # Scalar function space for 1D uniaxial stress
#     V = df.fem.functionspace(mesh, ("Lagrange", 1))
#     u = df.fem.Function(V, name="Displacement")
#     v = df.fem.Function(V, name="Velocity")
    
#     # Linear elastic law for 1D uniaxial stress
#     law = LinearElasticityModel(
#         parameters={"E": E, "nu": nu},
#         constraint=StressStrainConstraint.UNIAXIAL_STRESS,
#     )
    
#     # Boundary conditions
#     def left(x):
#         return np.isclose(x[0], 0.0)
    
#     def right(x):
#         return np.isclose(x[0], L)
    
#     # Fixed at left (x=0)
#     dofs_left = df.fem.locate_dofs_geometrical(V, left)
#     bc_left = df.fem.dirichletbc(df.fem.Constant(mesh, 0.0), dofs_left, V)
    
#     # Applied displacement at right (x=L)
#     dofs_right = df.fem.locate_dofs_geometrical(V, right)
#     bc_right = df.fem.dirichletbc(df.fem.Constant(mesh, u_applied), dofs_right, V)
    
#     bcs = [bc_left, bc_right]
    
#     # Create problem
#     problem = IncrSmallStrainProblem(law, u, bcs, q_degree=1)
    
#     # Create CDM solver with safety factor
#     safety_factor = 0.8
#     solver = CDMSolver(problem, density=rho, v=v, safety_factor=safety_factor)
    
#     # Time stepping
#     n_steps = int(T_end / problem.sim_time.dt)
#     print(n_steps)
#     for i in range(n_steps):
#         solver.step()
#         solver.problem.update()

        
#         # Apply boundary conditions after each step
#         #v.x.array[dofs_left] = 0.0
#         #v.x.array[dofs_right] = 0.0
#         #v.x.scatter_forward()
        
#         # Enforce displacement BCs
#         #problem.incr_disp.current.x.array[dofs_left] = 0.0
#         #problem.incr_disp.current.x.array[dofs_right] = u_applied
#         #problem.incr_disp.current.x.scatter_forward()
#     # Get numerical solution
#     u_numerical = problem.incr_disp.current.x.array.copy()
#     x_coords = V.tabulate_dof_coordinates()[:, 0]
    
#     # Compute analytical solution
#     t_final = n_steps * problem.sim_time.dt
#     u_analytical = analytical_1d_wave_solution(x_coords, t_final, L, c, u_applied)
    
#     # Check boundary conditions are satisfied
#     left_mask = np.isclose(x_coords, 0.0)
#     right_mask = np.isclose(x_coords, L)
#     assert np.allclose(u_numerical[left_mask], 0.0, atol=1e-10), "Left BC not satisfied"
#     assert np.allclose(u_numerical[right_mask], u_applied, atol=1e-10), "Right BC not satisfied"
    
#     # Compare with analytical solution (allow some error due to discretization)
#     error = np.abs(u_numerical - u_analytical)
#     max_error = np.max(error)
#     rel_error = max_error / u_applied
    
#     # Check that relative error is reasonable (< 30% for coarse mesh)
#     assert rel_error < 0.3, f"Relative error {rel_error:.2%} too large"


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
    u = df.fem.Function(V, name="Displacement")
    
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
    
    bc_left = df.fem.dirichletbc(df.fem.Constant(mesh, 0.0), dofs_left, V)
    bc_right = df.fem.dirichletbc(df.fem.Constant(mesh, 0.0), dofs_right, V)
    bcs = [bc_left, bc_right]
    
    # Create problem
    problem = IncrSmallStrainProblem(law, u, bcs, q_degree=1)
    
    # Set initial displacement (1st mode shape)
    x_coords = V.tabulate_dof_coordinates()[:, 0]
    u_initial = amplitude * np.sin(mode * np.pi * x_coords / L)
    
    # Create initial displacement function
    u0 = df.fem.Function(V, name="InitialDisplacement")
    u0.x.array[:] = u_initial
    # Ensure BCs are satisfied in initial condition
    u0.x.array[dofs_left] = 0.0
    u0.x.array[dofs_right] = 0.0
    u0.x.scatter_forward()
    
    # Create initial velocity function (zero)
    v0 = df.fem.Function(V, name="InitialVelocity")
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


# def test_cdm_energy_stability_1d():
#     """
#     Test stability for an undamped 1D system - displacements should stay bounded.
#     """
#     E = 210e9
#     nu = 0.3
#     rho = 7800
#     L = 1.0
#     n_elements = 20
    
#     mesh = df.mesh.create_interval(MPI.COMM_WORLD, n_elements, [0.0, L])
#     V = df.fem.functionspace(mesh, ("Lagrange", 1))
#     u = df.fem.Function(V, name="Displacement")
#     v = df.fem.Function(V, name="Velocity")
    
#     law = LinearElasticityModel(
#         parameters={"E": E, "nu": nu},
#         constraint=StressStrainConstraint.UNIAXIAL_STRESS,
#     )
    
#     def left(x):
#         return np.isclose(x[0], 0.0)
    
#     def right(x):
#         return np.isclose(x[0], L)
    
#     dofs_left = df.fem.locate_dofs_geometrical(V, left)
#     dofs_right = df.fem.locate_dofs_geometrical(V, right)
    
#     bc_left = df.fem.dirichletbc(df.fem.Constant(mesh, 0.0), dofs_left, V)
#     bc_right = df.fem.dirichletbc(df.fem.Constant(mesh, 0.0), dofs_right, V)
#     bcs = [bc_left, bc_right]
    
#     problem = IncrSmallStrainProblem(law, u, bcs, q_degree=2)
    
#     # Set initial displacement (sine wave)
#     x_coords = V.tabulate_dof_coordinates()[:, 0]
#     initial_amplitude = 0.0001
#     u_initial = initial_amplitude * np.sin(np.pi * x_coords / L)
#     problem.incr_disp.current.x.array[:] = u_initial
#     problem.incr_disp.current.x.scatter_forward()
    
#     # Apply boundary conditions
#     problem.incr_disp.current.x.array[dofs_left] = 0.0
#     problem.incr_disp.current.x.array[dofs_right] = 0.0
#     problem.incr_disp.current.x.scatter_forward()
    
#     problem.form_without_petsc(evaluate_tangent=False)
    
#     solver = CDMSolver(problem, rho, v=v, safety_factor=0.8)
    
#     n_steps = 100
#     for _ in range(n_steps):
#         solver.step()
        
#         v.x.array[dofs_left] = 0.0
#         v.x.array[dofs_right] = 0.0
#         v.x.scatter_forward()
        
#         problem.incr_disp.current.x.array[dofs_left] = 0.0
#         problem.incr_disp.current.x.array[dofs_right] = 0.0
#         problem.incr_disp.current.x.scatter_forward()
    
#     # Check displacements are bounded (stability)
#     max_disp = np.max(np.abs(problem.incr_disp.current.x.array))
#     assert max_disp < 10 * initial_amplitude, f"Displacement too large: {max_disp}"

if __name__=="__main__":
    test_cdm_1d_free_vibration(20)