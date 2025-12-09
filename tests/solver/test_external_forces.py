from __future__ import annotations

import dolfinx as df
import numpy as np
import ufl
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI

from fenics_constitutive.models import LinearElasticityModel, StressStrainConstraint
from fenics_constitutive.postprocessing import norm
from fenics_constitutive.solver import IncrSmallStrainProblem

youngs_modulus = 42.0
poissons_ratio = 0.3


def test_body_force_3d():
    """Test body force application on a unit cube."""
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)
    V = df.fem.functionspace(mesh, ("CG", 1, (3,)))
    u = df.fem.Function(V)
    
    law = LinearElasticityModel(
        parameters={"E": youngs_modulus, "nu": poissons_ratio},
        constraint=StressStrainConstraint.FULL,
    )

    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    # Fix the left boundary in all directions
    dofs_left = df.fem.locate_dofs_geometrical(V, left_boundary)
    bc_left = df.fem.dirichletbc(np.array([0.0, 0.0, 0.0]), dofs_left, V)

    # Define body force (gravity-like force in negative z-direction)
    v = ufl.TestFunction(V)
    body_force = df.fem.Constant(mesh, np.array([0.0, 0.0, -1.0]))
    external_forces_form = ufl.dot(body_force, v) * ufl.dx

    # Create problem with body force
    problem = IncrSmallStrainProblem(
        law,
        u,
        [bc_left],
        q_degree=2,
        external_forces=external_forces_form,
    )

    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    n, converged = solver.solve(u)
    
    assert converged, "Solver did not converge"
    
    # Check that displacement is non-zero (body force should cause deformation)
    u_norm = norm(u, problem.dxm)
    assert u_norm > 1e-10, f"Expected non-zero displacement, got {u_norm}"
    
    # Check that stress is non-zero
    stress_norm = norm(problem.stress_1, problem.dxm)
    assert stress_norm > 1e-10, f"Expected non-zero stress, got {stress_norm}"


def test_neumann_bc_3d():
    """Test Neumann boundary condition on a unit cube."""
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)
    V = df.fem.functionspace(mesh, ("CG", 1, (3,)))
    u = df.fem.Function(V)
    
    law = LinearElasticityModel(
        parameters={"E": youngs_modulus, "nu": poissons_ratio},
        constraint=StressStrainConstraint.FULL,
    )

    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    def right_boundary(x):
        return np.isclose(x[0], 1.0)

    # Fix the left boundary
    dofs_left = df.fem.locate_dofs_geometrical(V, left_boundary)
    bc_left = df.fem.dirichletbc(np.array([0.0, 0.0, 0.0]), dofs_left, V)

    # Apply surface traction on the right boundary (Neumann BC)
    v = ufl.TestFunction(V)
    traction = df.fem.Constant(mesh, np.array([1.0, 0.0, 0.0]))
    
    # Create facet tags for the right boundary
    facet_dim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(facet_dim, mesh.topology.dim)
    boundary_facets = df.mesh.locate_entities_boundary(mesh, facet_dim, right_boundary)
    facet_tags = df.mesh.meshtags(mesh, facet_dim, boundary_facets, np.full(len(boundary_facets), 1, dtype=np.int32))
    
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    external_forces_form = ufl.dot(traction, v) * ds(1)

    # Create problem with Neumann BC
    problem = IncrSmallStrainProblem(
        law,
        u,
        [bc_left],
        q_degree=2,
        external_forces=external_forces_form,
    )

    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    n, converged = solver.solve(u)
    
    assert converged, "Solver did not converge"
    
    # Check that displacement is non-zero (traction should cause deformation)
    u_norm = norm(u, problem.dxm)
    assert u_norm > 1e-10, f"Expected non-zero displacement, got {u_norm}"
    
    # Check that displacement in x-direction is positive (traction pulls in +x direction)
    max_u_x = MPI.COMM_WORLD.allreduce(np.max(u.x.array[::3]), MPI.MAX)
    assert max_u_x > 1e-10, f"Expected positive displacement in x-direction, got {max_u_x}"


def test_combined_forces_3d():
    """Test combination of body force and Neumann BC on a unit cube."""
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)
    V = df.fem.functionspace(mesh, ("CG", 1, (3,)))
    u = df.fem.Function(V)
    
    law = LinearElasticityModel(
        parameters={"E": youngs_modulus, "nu": poissons_ratio},
        constraint=StressStrainConstraint.FULL,
    )

    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    def right_boundary(x):
        return np.isclose(x[0], 1.0)

    # Fix the left boundary
    dofs_left = df.fem.locate_dofs_geometrical(V, left_boundary)
    bc_left = df.fem.dirichletbc(np.array([0.0, 0.0, 0.0]), dofs_left, V)

    # Define both body force and surface traction
    v = ufl.TestFunction(V)
    body_force = df.fem.Constant(mesh, np.array([0.0, 0.0, -0.5]))
    traction = df.fem.Constant(mesh, np.array([0.5, 0.0, 0.0]))
    
    # Create facet tags for the right boundary
    facet_dim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(facet_dim, mesh.topology.dim)
    boundary_facets = df.mesh.locate_entities_boundary(mesh, facet_dim, right_boundary)
    facet_tags = df.mesh.meshtags(mesh, facet_dim, boundary_facets, np.full(len(boundary_facets), 1, dtype=np.int32))
    
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    external_forces_form = ufl.dot(body_force, v) * ufl.dx + ufl.dot(traction, v) * ds(1)

    # Create problem with combined forces
    problem = IncrSmallStrainProblem(
        law,
        u,
        [bc_left],
        q_degree=2,
        external_forces=external_forces_form,
    )

    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    n, converged = solver.solve(u)
    
    assert converged, "Solver did not converge"
    
    # Check that displacement is non-zero
    u_norm = norm(u, problem.dxm)
    assert u_norm > 1e-10, f"Expected non-zero displacement, got {u_norm}"


def test_no_external_forces():
    """Test that the solver still works without external forces (backward compatibility)."""
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    V = df.fem.functionspace(mesh, ("CG", 1, (3,)))
    u = df.fem.Function(V)
    
    law = LinearElasticityModel(
        parameters={"E": youngs_modulus, "nu": poissons_ratio},
        constraint=StressStrainConstraint.FULL,
    )

    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    def right_boundary(x):
        return np.isclose(x[0], 1.0)

    dofs_left = df.fem.locate_dofs_geometrical(V, left_boundary)
    dofs_right = df.fem.locate_dofs_geometrical(V, right_boundary)
    bc_left = df.fem.dirichletbc(np.array([0.0, 0.0, 0.0]), dofs_left, V)
    bc_right = df.fem.dirichletbc(np.array([0.01, 0.0, 0.0]), dofs_right, V)

    # Create problem without external forces (default behavior)
    problem = IncrSmallStrainProblem(
        law,
        u,
        [bc_left, bc_right],
        q_degree=2,
    )

    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    n, converged = solver.solve(u)
    
    assert converged, "Solver did not converge"
    
    # Check basic displacement
    max_u = MPI.COMM_WORLD.allreduce(np.max(u.x.array), MPI.MAX)
    assert max_u > 0, "Expected positive displacement"


if __name__ == "__main__":
    test_body_force_3d()
    test_neumann_bc_3d()
    test_combined_forces_3d()
    test_no_external_forces()
