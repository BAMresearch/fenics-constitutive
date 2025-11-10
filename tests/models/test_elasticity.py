from __future__ import annotations

import dolfinx as df
import numpy as np
import pytest
import ufl
from dolfinx.nls.petsc import NewtonSolver

# from linear_elasticity_model import LinearElasticityModel
from mpi4py import MPI

from fenics_constitutive.models import (
    LinearElasticityModel,
    PlaneStrainFrom3D,
    StressStrainConstraint,
    UniaxialStrainFrom3D,
)
from fenics_constitutive.models.rust_models import LinearElasticity3D
from fenics_constitutive.postprocessing import norm
from fenics_constitutive.solver import IncrSmallStrainProblem

youngs_modulus = 42.0
poissons_ratio = 0.3


def test_uniaxial_stress():
    mesh = df.mesh.create_unit_interval(MPI.COMM_WORLD, 10)
    V = df.fem.functionspace(mesh, ("CG", 1))
    u = df.fem.Function(V)
    law = LinearElasticityModel(
        parameters={"E": youngs_modulus, "nu": poissons_ratio},
        constraint=StressStrainConstraint.UNIAXIAL_STRESS,
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
    n, converged = solver.solve(u)

    # Compare the result with the analytical solution
    diff = problem.stress_1 - ufl.as_vector(
        [
            youngs_modulus * 0.01,
        ]
    )
    assert norm(diff, problem.dxm) < 1e-10 / youngs_modulus * 0.01

    problem.update()
    # Check that the stress is updated correctly
    diff = problem.stress_0 - ufl.as_vector(
        [
            youngs_modulus * 0.01,
        ]
    )
    assert norm(diff, problem.dxm) < 1e-10 / youngs_modulus * 0.01

    # Check that the displacement is updated correctly
    max_u = MPI.COMM_WORLD.allreduce(np.max(problem._u0.x.array), MPI.MAX)
    assert max_u == displacement.value

    displacement.value = 0.02
    n, converged = solver.solve(u)

    # Compare the result of the updated problem with new BC with the analytical solution
    diff = problem.stress_1 - ufl.as_vector(
        [
            youngs_modulus * 0.02,
        ]
    )
    assert norm(diff, problem.dxm) < 1e-10 / youngs_modulus * 0.02


@pytest.mark.parametrize(
    ("factor"),
    [
        (0.5),
        (2.0),
        (3.0),
        (4.0),
    ],
)
def test_uniaxial_stress_two_laws(factor: float):
    mesh = df.mesh.create_unit_interval(MPI.COMM_WORLD, 2)
    V = df.fem.functionspace(mesh, ("CG", 1))
    u = df.fem.Function(V)
    cells_local = mesh.topology.index_map(mesh.topology.dim).global_to_local(
        np.arange(2, dtype=np.int32)
    )
    laws = [
        (
            LinearElasticityModel(
                parameters={"E": youngs_modulus, "nu": poissons_ratio},
                constraint=StressStrainConstraint.UNIAXIAL_STRESS,
            ),
            cells_local[0:1],
        ),
        (
            LinearElasticityModel(
                parameters={"E": factor * youngs_modulus, "nu": poissons_ratio},
                constraint=StressStrainConstraint.UNIAXIAL_STRESS,
            ),
            cells_local[1:2],
        ),
    ]

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
        laws,
        u,
        [bc_left, bc_right],
        1,
    )

    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    n, converged = solver.solve(u)
    problem.update()

    # Is the stress homogenous?
    assert abs(problem.stress_0.x.array[0] - problem.stress_0.x.array[1]) < 1e-10 / abs(
        problem.stress_0.x.array[0]
    )

    # Does the stiffer element have a proportionally lower strain?
    assert abs(
        problem._del_grad_u[0].x.array[0] - factor * problem._del_grad_u[1].x.array[0]
    ) < 1e-10 / abs(problem._del_grad_u[0].x.array[0])


def test_uniaxial_strain():
    mesh = df.mesh.create_unit_interval(MPI.COMM_WORLD, 2)
    V = df.fem.functionspace(mesh, ("CG", 1))
    u = df.fem.Function(V)
    law = LinearElasticityModel(
        parameters={"E": youngs_modulus, "nu": poissons_ratio},
        constraint=StressStrainConstraint.UNIAXIAL_STRAIN,
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
    n, converged = solver.solve(u)
    problem.update()

    analytical_stress = (
        youngs_modulus
        * (1.0 - poissons_ratio)
        / ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio))
    ) * displacement.value
    diff = problem.stress_0 - ufl.as_vector(
        [
            analytical_stress,
        ]
    )
    assert norm(diff, problem.dxm) < 1e-10 / analytical_stress

    # test the converter from 3D model to uniaxial strain model
    law_3d = LinearElasticityModel(
        parameters={"E": youngs_modulus, "nu": poissons_ratio},
        constraint=StressStrainConstraint.FULL,
    )
    wrapped_1d_law = UniaxialStrainFrom3D(law_3d)
    u_3d = df.fem.Function(V)
    problem_3d = IncrSmallStrainProblem(
        wrapped_1d_law,
        u_3d,
        [bc_left, bc_right],
        1,
    )
    solver_3d = NewtonSolver(MPI.COMM_WORLD, problem_3d)
    n, converged = solver_3d.solve(u_3d)
    problem_3d.update()

    # test that sigma_11 is the same as the analytical solution
    diff = problem_3d.stress_0 - ufl.as_vector(
        [
            analytical_stress,
        ]
    )
    assert norm(diff, problem_3d.dxm) < 1e-10 / analytical_stress

    # test that the stresses of the problem with uniaxial strain model
    # are the same as the stresses of the problem with the converted 3D model
    diff = problem_3d.stress_0 - problem.stress_0
    assert norm(diff, problem_3d.dxm) < 1e-10 / norm(problem.stress_0, problem.dxm)

    # test that the shear stresses are zero. Since this is uniaxial strain, the
    # stress can have nonzero \sigma_22 and \sigma_33 components
    assert np.linalg.norm(wrapped_1d_law.stress_3d[3:6]) < 1e-14
    # test that the displacement is the same in both versions
    diff = problem_3d._u - problem._u
    assert norm(diff, problem_3d.dxm) < 1e-14 / norm(problem._u, problem.dxm)


def test_plane_strain():
    # sanity check if out of plane stress is NOT zero
    mesh = df.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    V = df.fem.functionspace(mesh, ("CG", 1, (2,)))
    u = df.fem.Function(V)
    law = LinearElasticityModel(
        parameters={"E": youngs_modulus, "nu": poissons_ratio},
        constraint=StressStrainConstraint.PLANE_STRAIN,
    )

    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    def right_boundary(x):
        return np.isclose(x[0], 1.0)

    dofs_left = df.fem.locate_dofs_geometrical(V, left_boundary)
    dofs_right = df.fem.locate_dofs_geometrical(V, right_boundary)
    bc_left = df.fem.dirichletbc(np.array([0.0, 0.0]), dofs_left, V)
    bc_right = df.fem.dirichletbc(np.array([0.01, 0.0]), dofs_right, V)

    problem = IncrSmallStrainProblem(
        law,
        u,
        [bc_left, bc_right],
        1,
    )

    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    n, converged = solver.solve(u)
    problem.update()
    # test that the stress is nonzero in 33 direction
    assert norm(problem.stress_0[2], problem.dxm) > 1e-2

    # test the model conversion from 3D to plane strain
    law_3d = LinearElasticityModel(
        parameters={"E": youngs_modulus, "nu": poissons_ratio},
        constraint=StressStrainConstraint.FULL,
    )
    wrapped_2d_law = PlaneStrainFrom3D(law_3d)
    u_3d = df.fem.Function(V)
    problem_3d = IncrSmallStrainProblem(
        wrapped_2d_law,
        u_3d,
        [bc_left, bc_right],
        1,
    )
    solver_3d = NewtonSolver(MPI.COMM_WORLD, problem_3d)
    n, converged = solver_3d.solve(u_3d)
    problem_3d.update()
    # test that the stress is nonzero in 33 direction
    assert norm(problem_3d.stress_0[2], problem.dxm) > 1e-2

    # test that the displacement is the same in both versions
    diff = problem_3d._u - problem._u
    assert norm(diff, problem.dxm) / norm(problem._u, problem.dxm) < 1e-14
    # test that the stresses are the same in both versions
    diff = problem_3d.stress_0 - problem.stress_0
    assert norm(diff, problem.dxm) / norm(problem.stress_0, problem.dxm) < 1e-10


def test_plane_stress():
    # just a sanity check if out of plane stress is really zero
    mesh = df.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    V = df.fem.functionspace(mesh, ("CG", 1, (2,)))
    u = df.fem.Function(V)
    law = LinearElasticityModel(
        parameters={"E": youngs_modulus, "nu": poissons_ratio},
        constraint=StressStrainConstraint.PLANE_STRESS,
    )

    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    def right_boundary(x):
        return np.isclose(x[0], 1.0)

    # displacement = df.fem.Constant(mesh_2d, np.ndarray([0.01,0.0]))
    dofs_left = df.fem.locate_dofs_geometrical(V, left_boundary)
    dofs_right = df.fem.locate_dofs_geometrical(V, right_boundary)
    bc_left = df.fem.dirichletbc(np.array([0.0, 0.0]), dofs_left, V)
    bc_right = df.fem.dirichletbc(np.array([0.01, 0.0]), dofs_right, V)

    problem = IncrSmallStrainProblem(
        law,
        u,
        [bc_left, bc_right],
        1,
    )

    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    n, converged = solver.solve(u)
    problem.update()
    # test that the out of plane stress is zero
    assert norm(problem.stress_0[2], problem.dxm) < 1e-10

@pytest.mark.parametrize("model", [(LinearElasticityModel),(LinearElasticity3D)], ids=["python", "rust"])
def test_3d(model):
    # test the 3d case against a pure fenics implementation
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    V = df.fem.functionspace(mesh, ("CG", 1, (3,)))
    u = df.fem.Function(V)
    try:
        law = model(
            parameters={"E": youngs_modulus, "nu": poissons_ratio},
            constraint=StressStrainConstraint.FULL,
        )
    except:
        mu = youngs_modulus/(2*(1+poissons_ratio))
        kappa = youngs_modulus /(3*(1-2*poissons_ratio))
        print("mu python", mu)
        print("kappa python", kappa)

        law = model({"mu": np.array([mu]), "kappa": np.array([kappa])})
        #print(law.history_dim)
    
    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    def right_boundary(x):
        return np.isclose(x[0], 1.0)

    # displacement = df.fem.Constant(mesh_2d, np.ndarray([0.01,0.0]))
    dofs_left = df.fem.locate_dofs_geometrical(V, left_boundary)
    dofs_right = df.fem.locate_dofs_geometrical(V, right_boundary)
    bc_left = df.fem.dirichletbc(np.array([0.0, 0.0, 0.0]), dofs_left, V)
    bc_right = df.fem.dirichletbc(np.array([0.01, 0.0, 0.0]), dofs_right, V)

    problem = IncrSmallStrainProblem(
        law,
        u,
        [bc_left, bc_right],
        1,
    )

    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    n, converged = solver.solve(u)
    problem.update()
    v_, u_ = ufl.TestFunction(V), ufl.TrialFunction(V)

    def eps(v):
        return ufl.sym(ufl.grad(v))

    def sigma(v):
        eps = ufl.sym(ufl.grad(v))
        lam, mu = (
            youngs_modulus
            * poissons_ratio
            / ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio)),
            youngs_modulus / (2.0 * (1.0 + poissons_ratio)),
        )
        return 2.0 * mu * eps + lam * ufl.tr(eps) * ufl.Identity(len(v))

    zero = df.fem.Constant(mesh, np.array([0.0, 0.0, 0.0]))
    a = ufl.inner(ufl.grad(v_), sigma(u_)) * ufl.dx
    L = ufl.dot(zero, v_) * ufl.dx

    u_fenics = u.copy()
    problem_fenics = df.fem.petsc.LinearProblem(a, L, [bc_left, bc_right], u=u_fenics)
    problem_fenics.solve()

    # Check that the solution is the same
    diff = u_fenics - u
    assert norm(diff, problem.dxm) < 1e-8 / norm(u_fenics, problem.dxm)


if __name__ == "__main__":
    test_uniaxial_stress()
    test_uniaxial_strain()
    test_plane_stress()
    test_plane_strain()
    test_3d()
if __name__ == "__main__":
    test_uniaxial_stress()
    test_uniaxial_strain()
    test_plane_stress()
    test_plane_strain()
    test_3d()
