from mpi4py import MPI
import dolfinx as df
from linear_elasticity_model import LinearElasticityModel
from fenics_constitutive import Constraint, IncrSmallStrainProblem
import numpy as np
from dolfinx.nls.petsc import NewtonSolver

youngs_modulus = 42.
poissons_ratio = 0.3

def test_uniaxial_stress():
    mesh = df.mesh.create_unit_interval(MPI.COMM_WORLD, 2)
    V = df.fem.FunctionSpace(mesh, ("CG", 1))
    u = df.fem.Function(V)
    law = LinearElasticityModel(
        parameters={"E": youngs_modulus, "nu": poissons_ratio},
        constraint=Constraint.UNIAXIAL_STRESS,
    )

    def left_boundary(x):
        return np.isclose(x[0], 0.0)
    
    def right_boundary(x):
        return np.isclose(x[0], 1.0)
    
    displacement = df.fem.Constant(mesh, 0.01)
    dofs_left = df.fem.locate_dofs_geometrical(V, left_boundary)
    dofs_right = df.fem.locate_dofs_geometrical(V, right_boundary)
    bc_left = df.fem.dirichletbc(df.fem.Constant(mesh,0.0), dofs_left, V)
    bc_right = df.fem.dirichletbc(displacement, dofs_right, V)

    
    problem = IncrSmallStrainProblem(
        law,
        u,
        [bc_left, bc_right],
    )

    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    n, converged = solver.solve(u)
    assert abs(problem.stress_1.x.array[0] - youngs_modulus * 0.01) < 1e-10 / (youngs_modulus * 0.01)

    problem.update()
    assert abs(problem.stress_0.x.array[0] - youngs_modulus * 0.01) < 1e-10 / (youngs_modulus * 0.01)
    assert np.max(problem._u0.x.array) == displacement.value

    displacement.value = 0.02
    n, converged = solver.solve(u)
    assert abs(problem.stress_1.x.array[0] - youngs_modulus * 0.02) < 1e-10 / (youngs_modulus * 0.02)
