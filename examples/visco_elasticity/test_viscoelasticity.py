from __future__ import annotations

import dolfinx as df
import numpy as np
import pytest
import ufl
from dolfinx.nls.petsc import NewtonSolver
from spring_kelvin_model import SpringKelvinModel
from mpi4py import MPI

from fenics_constitutive import Constraint, IncrSmallStrainProblem

youngs_modulus = 42.0
poissons_ratio = 0.3
visco_modulus = 10.0
relaxation_time = 10.0


def test_relaxation_uniaxial_stress():
    mesh = df.mesh.create_unit_interval(MPI.COMM_WORLD, 2)
    V = df.fem.FunctionSpace(mesh, ("CG", 1))
    u = df.fem.Function(V)
    law = SpringKelvinModel(
        parameters={"E0": youngs_modulus, "E1": visco_modulus, "tau": relaxation_time, "nu": poissons_ratio},
        constraint=Constraint.UNIAXIAL_STRESS,
    )

    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    def right_boundary(x):
        return np.isclose(x[0], 1.0)

    displacement = df.fem.Constant(mesh, 0.01)
    dofs_left = df.fem.locate_dofs_geometrical(V, left_boundary)
    dofs_right = df.fem.locate_dofs_geometrical(V, right_boundary)
    bc_left = df.fem.dirichletbc(df.fem.Constant(mesh, 0.0), dofs_left, V)
    bc_right = df.fem.dirichletbc(displacement, dofs_right, V)

    problem = IncrSmallStrainProblem(
        law,
        u,
        [bc_left, bc_right],
        1,
    )

    solver = NewtonSolver(MPI.COMM_WORLD, problem)

    time = [0]
    disp = []
    stress = []

    # elastic first step
    problem._time = 0
    solver.solve(u)
    problem.update()
    disp.append(u.x.array[-1])
    stress.append(problem.stress_1.x.array[-1])

    # set time step and solve until total time
    dt = 2
    problem._time = dt
    total_time = 10*relaxation_time
    while time[-1] < total_time:
        time.append(time[-1]+dt)
        niter, converged = solver.solve(u)
        problem.update()
        print(f"time {time} Converged: {converged} in {niter} iterations.")

        # print(problem.stress_1.x.array)  # mandel stress at time t
        # print(u.x.array)
        disp.append(u.x.array[-1])
        stress.append(problem.stress_1.x.array[-1])

    #analytic solution for 1D Kelvin model
    stress_0_ana = youngs_modulus * displacement.value/1.
    stress_final_ana = youngs_modulus * visco_modulus / (youngs_modulus + visco_modulus) * displacement.value/1.

    assert abs(stress[0] - stress_0_ana) < 1e-8
    assert abs(stress[-1] - stress_final_ana) < 1e-8



if __name__ == "__main__":
    test_relaxation_uniaxial_stress()