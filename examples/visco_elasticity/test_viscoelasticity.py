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


def test_creep_uniaxial_stress():
    mesh = df.mesh.create_unit_interval(MPI.COMM_WORLD, 2)
    V = df.fem.FunctionSpace(mesh, ("CG", 1))
    u = df.fem.Function(V)
    law = SpringKelvinModel(
        parameters={"E0": youngs_modulus, "E1": visco_modulus, "tau": relaxation_time, "nu": poissons_ratio},
        constraint=Constraint.UNIAXIAL_STRESS,
    )
    # law = LinearElasticityModel(
    #     parameters={"E": youngs_modulus, "nu": poissons_ratio},
    #     constraint=Constraint.UNIAXIAL_STRESS,
    # )

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
    )

    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    n, converged = solver.solve(u)
    # assert abs(problem.stress_1.x.array[0] - youngs_modulus * 0.01) < 1e-10 / (
    #     youngs_modulus * 0.01
    # )
    #
    # problem.update()
    # assert abs(problem.stress_0.x.array[0] - youngs_modulus * 0.01) < 1e-10 / (
    #     youngs_modulus * 0.01
    # )
    # assert np.max(problem._u0.x.array) == displacement.value
    #
    # displacement.value = 0.02
    # n, converged = solver.solve(u)
    # assert abs(problem.stress_1.x.array[0] - youngs_modulus * 0.02) < 1e-10 / (
    #     youngs_modulus * 0.02
    # )


if __name__ == "__main__":
    test_creep_uniaxial_stress()