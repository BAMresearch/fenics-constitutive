"""
Tests for the critical timestep calculation in the CDM solver.

The critical timestep for the central difference method is:
    dt_crit = 2 / omega_max

where omega_max is the maximum natural frequency of the system.

For a simple 1D bar with uniaxial stress, the wave speed is c = sqrt(E/rho),
and the critical timestep scales as:
    dt_crit ~ h / c = h * sqrt(rho / E)

This means:
    - dt_crit is proportional to h (element size)
    - dt_crit is proportional to sqrt(rho) (density)
    - dt_crit is proportional to 1/sqrt(E) (stiffness)
"""

from __future__ import annotations

import dolfinx as df
import numpy as np
import pytest
from mpi4py import MPI

from fenics_constitutive.models.linear_elasticity_model import LinearElasticityModel
from fenics_constitutive.models.interfaces import StressStrainConstraint
from fenics_constitutive.solver.central_difference_method import critical_timestep


def _create_1d_problem(
    n_elements: int, L: float, E: float, nu: float
) -> tuple[df.fem.Function, LinearElasticityModel, np.ndarray]:
    """Helper to create a 1D mesh, function space, and law."""
    mesh = df.mesh.create_interval(MPI.COMM_WORLD, n_elements, [0.0, L])
    V = df.fem.functionspace(mesh, ("Lagrange", 1))
    u = df.fem.Function(V)

    law = LinearElasticityModel(
        parameters={"E": E, "nu": nu},
        constraint=StressStrainConstraint.UNIAXIAL_STRESS,
    )

    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    return u, law, cells


class TestCriticalTimestepScaling:
    """Test that critical timestep scales correctly with physical parameters."""

    # Reference parameters
    E_ref = 210e9  # Young's modulus (Pa) - steel
    nu = 0.3  # Poisson's ratio
    rho_ref = 7800.0  # Density (kg/m^3)
    L = 1.0  # Bar length (m)
    n_elements = 10

    def test_scales_linearly_with_element_size(self):
        """
        dt_crit should scale linearly with element size h.
        
        dt_crit ~ h means that halving the element size halves the timestep.
        """
        h_values = [0.1, 0.05, 0.025]  # Different element sizes
        dt_values = []

        for h in h_values:
            n_elements = int(self.L / h)
            u, law, cells = _create_1d_problem(n_elements, self.L, self.E_ref, self.nu)

            laws = [(law, cells)]
            density = [self.rho_ref]
            dt = critical_timestep(laws, density, u)
            dt_values.append(dt[0])

        dt_values = np.array(dt_values)
        h_values = np.array(h_values)

        # dt / h should be approximately constant
        ratios = dt_values / h_values
        relative_variation = np.std(ratios) / np.mean(ratios)

        assert relative_variation < 0.05, (
            f"dt/h ratios vary too much: {ratios}, relative variation: {relative_variation:.2%}"
        )

    def test_scales_with_sqrt_density(self):
        """
        dt_crit should scale with sqrt(rho).
        
        Doubling density should increase dt_crit by sqrt(2) ≈ 1.414.
        """
        rho_values = [self.rho_ref, 2 * self.rho_ref, 4 * self.rho_ref]
        dt_values = []

        u, law, cells = _create_1d_problem(
            self.n_elements, self.L, self.E_ref, self.nu
        )

        for rho in rho_values:
            laws = [(law, cells)]
            density = [rho]
            dt = critical_timestep(laws, density, u)
            dt_values.append(dt[0])

        dt_values = np.array(dt_values)
        rho_values = np.array(rho_values)

        # dt / sqrt(rho) should be approximately constant
        ratios = dt_values / np.sqrt(rho_values)
        relative_variation = np.std(ratios) / np.mean(ratios)

        assert relative_variation < 0.05, (
            f"dt/sqrt(rho) ratios vary too much: {ratios}, relative variation: {relative_variation:.2%}"
        )

    def test_scales_with_inverse_sqrt_stiffness(self):
        """
        dt_crit should scale with 1/sqrt(E).
        
        Doubling stiffness should decrease dt_crit by sqrt(2) ≈ 1.414.
        """
        E_values = [self.E_ref, 2 * self.E_ref, 4 * self.E_ref]
        dt_values = []

        for E in E_values:
            u, law, cells = _create_1d_problem(self.n_elements, self.L, E, self.nu)

            laws = [(law, cells)]
            density = [self.rho_ref]
            dt = critical_timestep(laws, density, u)
            dt_values.append(dt[0])

        dt_values = np.array(dt_values)
        E_values = np.array(E_values)

        # dt * sqrt(E) should be approximately constant
        ratios = dt_values * np.sqrt(E_values)
        relative_variation = np.std(ratios) / np.mean(ratios)

        assert relative_variation < 0.05, (
            f"dt*sqrt(E) ratios vary too much: {ratios}, relative variation: {relative_variation:.2%}"
        )

    def test_analytical_value_1d(self):
        """
        For 1D uniaxial stress: dt_crit = h / c where c = sqrt(E/rho).
        """
        h = self.L / self.n_elements
        c = np.sqrt(self.E_ref / self.rho_ref)
        dt_expected = h / c

        u, law, cells = _create_1d_problem(
            self.n_elements, self.L, self.E_ref, self.nu
        )

        laws = [(law, cells)]
        density = [self.rho_ref]
        dt_computed = critical_timestep(laws, density, u)

        # Allow some tolerance due to numerical computation of eigenvalues
        rel_diff = abs(dt_computed[0] - dt_expected) / dt_expected
        print(dt_computed,dt_expected)
        assert rel_diff < 1e-10, (
            f"Critical timestep differs from analytical value by {rel_diff:.2%}"
        )

    def test_custom_h_parameter(self):
        """Test that providing a custom h parameter overrides mesh-based calculation."""
        u, law, cells = _create_1d_problem(
            self.n_elements, self.L, self.E_ref, self.nu
        )

        laws = [(law, cells)]
        density = [self.rho_ref]

        # Compute with automatic h
        dt_auto = critical_timestep(laws, density, u)

        # Compute with custom h (half the actual element size)
        h_custom = (self.L / self.n_elements) / 2
        dt_custom = critical_timestep(laws, density, u, h=h_custom)

        # dt should scale linearly with h, so dt_custom should be ~half of dt_auto
        ratio = dt_custom[0] / dt_auto[0]
        assert 0.45 < ratio < 0.55, (
            f"Custom h=h/2 should give dt ≈ dt_auto/2, but ratio is {ratio:.3f}"
        )

    def test_positive_timestep(self):
        """Critical timestep must always be positive."""
        u, law, cells = _create_1d_problem(
            self.n_elements, self.L, self.E_ref, self.nu
        )

        laws = [(law, cells)]
        density = [self.rho_ref]
        dt = critical_timestep(laws, density, u)

        assert dt[0] > 0, "Critical timestep must be positive"


class TestCriticalTimestep2D:
    """Test critical timestep for 2D problems."""

    E = 210e9
    nu = 0.3
    rho = 7800.0

    @pytest.mark.parametrize("constraint", [
        StressStrainConstraint.PLANE_STRESS,
        StressStrainConstraint.PLANE_STRAIN,
    ])
    def test_2d_quadrilateral(self, constraint):
        """Test critical timestep computation for 2D quadrilateral mesh."""
        mesh = df.mesh.create_rectangle(
            MPI.COMM_WORLD,
            [[0.0, 0.0], [1.0, 1.0]],
            [5, 5],
            cell_type=df.mesh.CellType.quadrilateral,
        )
        V = df.fem.functionspace(mesh, ("Lagrange", 1, (2,)))
        u = df.fem.Function(V)

        law = LinearElasticityModel(
            parameters={"E": self.E, "nu": self.nu},
            constraint=constraint,
        )

        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        cells = np.arange(0, num_cells, dtype=np.int32)

        laws = [(law, cells)]
        density = [self.rho]
        dt = critical_timestep(laws, density, u)

        assert dt[0] > 0, "Critical timestep must be positive"
        # For 2D, wave speed is higher (P-wave), so dt should be smaller than 1D estimate
        # Just verify it's in a reasonable range
        h = 1.0 / 5  # element size
        c_p = np.sqrt(self.E / self.rho)  # approximate (ignoring Poisson effect)
        dt_rough_estimate = 2 * h / c_p
        assert dt[0] < dt_rough_estimate, (
            "2D critical timestep should be smaller than 1D estimate due to P-wave"
        )


class TestCriticalTimestep3D:
    """Test critical timestep for 3D problems."""

    E = 210e9
    nu = 0.3
    rho = 7800.0

    def test_3d_hexahedron(self):
        """Test critical timestep computation for 3D hexahedral mesh."""
        mesh = df.mesh.create_box(
            MPI.COMM_WORLD,
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            [3, 3, 3],
            cell_type=df.mesh.CellType.hexahedron,
        )
        V = df.fem.functionspace(mesh, ("Lagrange", 1, (3,)))
        u = df.fem.Function(V)

        law = LinearElasticityModel(
            parameters={"E": self.E, "nu": self.nu},
            constraint=StressStrainConstraint.FULL,
        )

        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        cells = np.arange(0, num_cells, dtype=np.int32)

        laws = [(law, cells)]
        density = [self.rho]
        dt = critical_timestep(laws, density, u)

        assert dt[0] > 0, "Critical timestep must be positive"
