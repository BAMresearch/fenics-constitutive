"""
Tests for the diagonal mass matrix computation in the CDM solver.

We verify that the diagonal (lumped) mass matrix correctly represents the 
total mass when using multiple materials with different densities.
"""

from __future__ import annotations

import dolfinx as df
import numpy as np
import pytest
from mpi4py import MPI

from fenics_constitutive.models.interfaces import StressStrainConstraint
from fenics_constitutive.models.linear_elasticity_model import LinearElasticityModel
from fenics_constitutive.solver import IncrSmallStrainProblem
from fenics_constitutive.solver.central_difference_method import diagonal_inverted_mass


@pytest.mark.parametrize("n_elements", [4, 8])
def test_diagonal_mass_two_materials_hexahedron(n_elements: int):
    """
    Test that the diagonal mass matrix has the correct total mass for a cube
    with two different materials (one half with density rho1, other half with rho2).
    
    The total mass should be:
        M_total = rho1 * V1 + rho2 * V2
    where V1 and V2 are the volumes of each material region.
    
    For a unit cube split in half:
        M_total = rho1 * 0.5 + rho2 * 0.5
    """
    # Material parameters
    E = 210e9
    nu = 0.3
    rho1 = 7800.0  # Density material 1 (kg/m^3)
    rho2 = 2700.0  # Density material 2 (kg/m^3) - like aluminum
    
    # Create unit cube mesh
    mesh = df.mesh.create_box(
        MPI.COMM_WORLD,
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        [n_elements, n_elements, n_elements],
        cell_type=df.mesh.CellType.hexahedron,
    )
    
    # Vector function space for 3D elasticity
    V = df.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
    u = df.fem.Function(V, name="Displacement")
    
    # Linear elastic laws for both materials
    law1 = LinearElasticityModel(
        parameters={"E": E, "nu": nu},
        constraint=StressStrainConstraint.FULL,
    )
    law2 = LinearElasticityModel(
        parameters={"E": E, "nu": nu},
        constraint=StressStrainConstraint.FULL,
    )
    
    # Split cells by x-coordinate (x < 0.5 -> material 1, x >= 0.5 -> material 2)
    mesh.topology.create_connectivity(mesh.topology.dim, 0)
    cell_midpoints = df.mesh.compute_midpoints(mesh, mesh.topology.dim, 
                                                np.arange(mesh.topology.index_map(mesh.topology.dim).size_local, dtype=np.int32))
    
    cells_mat1 = np.where(cell_midpoints[:, 0] < 0.5)[0].astype(np.int32)
    cells_mat2 = np.where(cell_midpoints[:, 0] >= 0.5)[0].astype(np.int32)
    
    # Create problem with two laws
    laws = [(law1, cells_mat1), (law2, cells_mat2)]
    
    # Dummy boundary conditions (not used for mass computation)
    def left(x):
        return np.isclose(x[0], 0.0)
    
    dofs_left = df.fem.locate_dofs_geometrical(V, left)
    bc_left = df.fem.dirichletbc(
        df.fem.Constant(mesh, np.array([0.0, 0.0, 0.0])), dofs_left, V
    )
    bcs = [bc_left]
    
    problem = IncrSmallStrainProblem(laws, u, bcs, q_degree=1)
    
    # Compute diagonal inverted mass
    densities = [rho1, rho2]
    M_inv = diagonal_inverted_mass(V, densities, problem._law_on_submeshs)
    
    # The diagonal mass matrix satisfies: M @ ones = M_lumped
    # where M_lumped is the sum of each row of the consistent mass matrix
    # Total mass = sum of all diagonal entries = sum(1/M_inv^{-1}) = sum(M)
    # But M_inv contains 1/M_ii, so total mass contribution from each dof is 1/M_inv
    
    # Sum the diagonal mass (inverse of M_inv) to get total mass
    # Note: Each node has 3 DOFs (x,y,z), so we sum over all
    total_mass_numerical = np.sum(1.0 / M_inv.x.array)
    
    # Expected total mass: rho1 * V1 + rho2 * V2
    # For a unit cube split in half: V1 = V2 = 0.5
    expected_mass = rho1 * 0.5 + rho2 * 0.5
    
    # Allow some tolerance due to numerical integration
    rel_error = abs(total_mass_numerical - 3.0*expected_mass) / expected_mass
    assert rel_error < 0.01, f"Total mass error: {rel_error:.2%}, got {total_mass_numerical}, expected {expected_mass}"


@pytest.mark.parametrize("n_elements", [4, 8])
def test_diagonal_mass_uniform_density_hexahedron(n_elements: int):
    """
    Test that the diagonal mass matrix has the correct total mass for a cube
    with uniform density.
    
    The total mass should be: M_total = rho * V = rho * 1.0 (unit cube)
    """
    E = 210e9
    nu = 0.3
    rho = 7800.0
    
    mesh = df.mesh.create_box(
        MPI.COMM_WORLD,
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        [n_elements, n_elements, n_elements],
        cell_type=df.mesh.CellType.hexahedron,
    )
    
    V = df.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
    u = df.fem.Function(V, name="Displacement")
    
    law = LinearElasticityModel(
        parameters={"E": E, "nu": nu},
        constraint=StressStrainConstraint.FULL,
    )
    
    def left(x):
        return np.isclose(x[0], 0.0)
    
    dofs_left = df.fem.locate_dofs_geometrical(V, left)
    bc_left = df.fem.dirichletbc(
        df.fem.Constant(mesh, np.array([0.0, 0.0, 0.0])), dofs_left, V
    )
    bcs = [bc_left]
    
    problem = IncrSmallStrainProblem(law, u, bcs, q_degree=1)
    
    M_inv = diagonal_inverted_mass(V, [rho], problem._law_on_submeshs)
    
    total_mass_numerical = np.sum(1.0 / M_inv.x.array)
    expected_mass = rho * 1.0  # rho * volume of unit cube
    
    rel_error = abs(total_mass_numerical - 3.0 * expected_mass) / expected_mass
    assert rel_error < 0.01, f"Total mass error: {rel_error:.2%}, got {total_mass_numerical}, expected {expected_mass}"


@pytest.mark.parametrize("n_elements", [4, 8])
def test_diagonal_mass_two_materials_quadrilateral(n_elements: int):
    """
    Test the diagonal mass matrix for a 2D quadrilateral mesh with two materials.
    
    For a unit square split in half with plane stress:
        M_total = rho1 * A1 + rho2 * A2 (per unit thickness)
    """
    E = 210e9
    nu = 0.3
    rho1 = 7800.0
    rho2 = 2700.0
    
    mesh = df.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[0.0, 0.0], [1.0, 1.0]],
        [n_elements, n_elements],
        cell_type=df.mesh.CellType.quadrilateral,
    )
    
    V = df.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
    u = df.fem.Function(V, name="Displacement")
    
    law1 = LinearElasticityModel(
        parameters={"E": E, "nu": nu},
        constraint=StressStrainConstraint.PLANE_STRESS,
    )
    law2 = LinearElasticityModel(
        parameters={"E": E, "nu": nu},
        constraint=StressStrainConstraint.PLANE_STRESS,
    )
    
    mesh.topology.create_connectivity(mesh.topology.dim, 0)
    cell_midpoints = df.mesh.compute_midpoints(
        mesh, mesh.topology.dim,
        np.arange(mesh.topology.index_map(mesh.topology.dim).size_local, dtype=np.int32)
    )
    
    cells_mat1 = np.where(cell_midpoints[:, 0] < 0.5)[0].astype(np.int32)
    cells_mat2 = np.where(cell_midpoints[:, 0] >= 0.5)[0].astype(np.int32)
    
    laws = [(law1, cells_mat1), (law2, cells_mat2)]
    
    def left(x):
        return np.isclose(x[0], 0.0)
    
    dofs_left = df.fem.locate_dofs_geometrical(V, left)
    bc_left = df.fem.dirichletbc(
        df.fem.Constant(mesh, np.array([0.0, 0.0])), dofs_left, V
    )
    bcs = [bc_left]
    
    problem = IncrSmallStrainProblem(laws, u, bcs, q_degree=1)
    
    densities = [rho1, rho2]
    M_inv = diagonal_inverted_mass(V, densities, problem._law_on_submeshs)
    
    total_mass_numerical = np.sum(1.0 / M_inv.x.array)
    expected_mass = rho1 * 0.5 + rho2 * 0.5  # Area of each half
    
    rel_error = abs(total_mass_numerical - 2.0 * expected_mass) / expected_mass
    assert rel_error < 0.01, f"Total mass error: {rel_error:.2%}, got {total_mass_numerical}, expected {expected_mass}"
