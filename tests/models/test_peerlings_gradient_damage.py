from __future__ import annotations

import basix
import dolfinx as df
import numpy as np
import scipy.optimize
from basix import ElementFamily
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from sympy import N, Symbol, cos, exp, integrate, lambdify, symbols

from fenics_constitutive.models import StressStrainConstraint
from fenics_constitutive.models.peerlings_gradient_damage import (
    PeerlingsGradientPerfectDamage,
)
from fenics_constitutive.solver._gradient_enhanced_solver import (
    IncrSmallStrainGradientProblem,
)


class PeerlingsAnalytic:
    def __init__(
        self,
        L: float,
        W_half_percentage: int,
        deltaL: float,
        alpha: float,
        E: float,
        eps_0: float,
        l: float,
    ):
        assert W_half_percentage < 50
        W = W_half_percentage / 100.0 * 2 * L
        self.L, self.W, self.deltaL, self.alpha = (
            L,
            W,
            deltaL,
            alpha,
        )  # 100.0, 10.0, 0.05, 0.1
        self.E, self.kappa0, self.l = E, eps_0, l  # 20000.0, 1.0e-4, 1.0

        self._calculate_coeffs()

    def _calculate_coeffs(self):
        """
        The analytic solution is following Peerlings paper (1996) but with
        b(paper) = b^2 (here)
        g(paper) = g^2 (here)
        c(paper) = l^2 (here)
        This modification eliminates all the sqrts in the formulations.
        Plus: the formulation of the GDM in terms of l ( = sqrt(c) ) is
        more common in modern publications.
        """

        # imports only used here...

        # unknowns
        x = Symbol("x")
        unknowns = symbols("A1, A2, B1, B2, C, b, g, w")
        A1, A2, B1, B2, C, b, g, w = unknowns

        l = self.l
        kappa0 = self.kappa0

        # 0 <= x <= W/2
        e1 = C * cos(g / l * x)
        # W/2 <  x <= w/2
        e2 = B1 * exp(b / l * x) + B2 * exp(-b / l * x)
        # w/2 <  x <= L/2
        e3 = A1 * exp(x / l) + A2 * exp(-x / l) + (1 - b * b) * kappa0

        de1, de2, de3 = e1.diff(x), e2.diff(x), e3.diff(x)

        eq1 = N(e1.subs(x, self.W / 2) - e2.subs(x, self.W / 2))
        eq2 = N(de1.subs(x, self.W / 2) - de2.subs(x, self.W / 2))
        eq3 = N(e2.subs(x, w / 2) - kappa0)
        eq4 = N(de2.subs(x, w / 2) - de3.subs(x, w / 2))
        eq5 = N(e3.subs(x, w / 2) - kappa0)
        eq6 = N(de3.subs(x, self.L / 2))
        eq7 = N((1 - self.alpha) * (1 + g * g) - (1 - b * b))
        eq8 = N(
            integrate(e1, (x, 0, self.W / 2))
            + integrate(e2, (x, self.W / 2, w / 2))
            + integrate(e3, (x, w / 2, self.L / 2))
            - self.deltaL / 2
        )

        eqs = [
            lambdify(unknowns, eq) for eq in [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8]
        ]

        def global_func(x):
            return np.array([eqs[i](*x) for i in range(8)])

        result = scipy.optimize.root(
            global_func, [0.0, 5e2, 3e-7, 7e-3, 3e-3, 3e-1, 2e-1, 4e1]
        )
        if not result["success"]:
            msg = "Could not find the correct coefficients. Try to tweak the initial values."
            raise RuntimeError(msg)

        self.coeffs = result["x"]

    def e(self, x):
        A1, A2, B1, B2, C, b, g, w = self.coeffs
        if x <= self.W / 2.0:
            return C * np.cos(g / self.l * x)
        if x <= w / 2.0:
            return B1 * np.exp(b / self.l * x) + B2 * np.exp(-b / self.l * x)
        return (
            (1.0 - b * b) * self.kappa0
            + A1 * np.exp(x / self.l)
            + A2 * np.exp(-x / self.l)
        )


class PeerlingsNumeric:
    def __init__(
        self,
        L: float,
        W_half_percentage: int,
        deltaL: float,
        alpha: float,
        E: float,
        eps_0: float,
        l: float,
        h_refinement: int,
    ):
        assert W_half_percentage < 50
        W = W_half_percentage / 100.0 * 2 * L
        self.L, self.W, self.deltaL, self.alpha = (
            L,
            W,
            deltaL,
            alpha,
        )  # 100.0, 10.0, 0.05, 0.1
        self.E, self.kappa0, self.l = E, eps_0, l  # 20000.0, 1.0e-4, 1.0
        n_elements = 50 * h_refinement
        mesh = df.mesh.create_interval(
            MPI.COMM_WORLD, n_elements, np.array([0.0, L / 2.0])
        )

        element = basix.ufl.element(ElementFamily.P, basix.CellType.interval, 1)
        mixed_element = basix.ufl.mixed_element([element, element])
        mixed_space = df.fem.functionspace(mesh, mixed_element)
        u = df.fem.Function(mixed_space)

        law = PeerlingsGradientPerfectDamage(
            parameters={
                "E": self.E,
                "nu": 0.3,
                "eps_0": self.kappa0,
                "omega_max": 1.0,
            },  # nu does not do anythng in uniaxial stress
            constraint=StressStrainConstraint.UNIAXIAL_STRESS,
        )

        law_notch = PeerlingsGradientPerfectDamage(
            parameters={
                "E": self.E * (1.0 - self.alpha),
                "nu": 0.3,
                "eps_0": self.kappa0,
                "omega_max": 1.0,
            },  # nu does not do anythng in uniaxial stress
            constraint=StressStrainConstraint.UNIAXIAL_STRESS,
        )

        def left_boundary(x):
            return np.isclose(x[0], 0.0)
            # return np.isclose(x[0], -self.L/2.)

        def right_boundary(x):
            return np.isclose(x[0], self.L / 2.0)

        eps_mesh = (L / 2 / n_elements) * 1e-4

        def notch_marker(x):
            return x[0] <= self.W / 2.0 + eps_mesh
            # return (x[0]>=-self.W/2.0-eps_mesh) & (x[0]<=self.W/2.0+eps_mesh)

        def rest_marker(x):
            return x[0] >= self.W / 2.0 - eps_mesh
            # return (x[0]<=-self.W/2.0+eps_mesh) | (x[0]>=self.W/2.0-eps_mesh)

        cells_notch = df.mesh.locate_entities(mesh, mesh.topology.dim, notch_marker)
        cells_rest = df.mesh.locate_entities(mesh, mesh.topology.dim, rest_marker)
        print(cells_rest.shape, cells_notch.shape)
        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts

        assert cells_rest.size + cells_notch.size == num_cells

        displacement_left = df.fem.Constant(mesh, np.float64(0.0))
        displacement_right = df.fem.Constant(mesh, np.float64(self.deltaL / 2.0))

        entities_left = df.mesh.locate_entities_boundary(
            mixed_space.mesh, 0, left_boundary
        )
        entities_right = df.mesh.locate_entities_boundary(
            mixed_space.mesh, 0, right_boundary
        )
        dofs_left = df.fem.locate_dofs_topological(mixed_space.sub(0), 0, entities_left)
        dofs_right = df.fem.locate_dofs_topological(
            mixed_space.sub(0), 0, entities_right
        )
        bc_left = df.fem.dirichletbc(displacement_left, dofs_left, mixed_space.sub(0))
        bc_right = df.fem.dirichletbc(
            displacement_right, dofs_right, mixed_space.sub(0)
        )

        problem = IncrSmallStrainGradientProblem(
            [(law, cells_rest), (law_notch, cells_notch)],
            u,
            [bc_left, bc_right],
            1,
            self.l,
        )

        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        solver.rtol = 1e-8
        solver.atol = 1e-8
        # solver.maxit = 200
        # solver.criterion="incremental"
        displacements = np.linspace(0.0, displacement_right.value, 20)
        for d in displacements:
            displacement_right.value = d
            n, converged = solver.solve(u)
            print("converged in ", n, "iterations. displ:", displacement_right.value)
            problem.update()

        self.problem = problem


class PeerlingsNumericPlaneStrain:
    def __init__(
        self,
        L: float,
        W_half_percentage: int,
        deltaL: float,
        alpha: float,
        E: float,
        eps_0: float,
        l: float,
        h_refinement: int,
    ):
        assert W_half_percentage < 50
        W = W_half_percentage / 100.0 * 2 * L
        self.L, self.W, self.deltaL, self.alpha = (
            L,
            W,
            deltaL,
            alpha,
        )  # 100.0, 10.0, 0.05, 0.1
        self.E, self.kappa0, self.l = E, eps_0, l  # 20000.0, 1.0e-4, 1.0
        n_elements = 50 * h_refinement
        mesh = df.mesh.create_rectangle(
            MPI.COMM_WORLD,
            np.array([[0.0, 0.0], [L / 2.0, 1.0]]),
            [n_elements, 1],
            cell_type=df.mesh.CellType.quadrilateral,
        )

        element_d = basix.ufl.element(
            ElementFamily.P, basix.CellType.quadrilateral, 1, shape=(2,)
        )
        element_eps = basix.ufl.element(
            ElementFamily.P, basix.CellType.quadrilateral, 1
        )
        mixed_element = basix.ufl.mixed_element([element_d, element_eps])
        mixed_space = df.fem.functionspace(mesh, mixed_element)
        u = df.fem.Function(mixed_space)

        law = PeerlingsGradientPerfectDamage(
            parameters={
                "E": self.E,
                "nu": 0.3,
                "eps_0": self.kappa0,
                "omega_max": 1.0,
            },  # nu does not do anythng in uniaxial stress
            constraint=StressStrainConstraint.PLANE_STRAIN,
        )

        law_notch = PeerlingsGradientPerfectDamage(
            parameters={
                "E": self.E * (1.0 - self.alpha),
                "nu": 0.3,
                "eps_0": self.kappa0,
                "omega_max": 1.0,
            },  # nu does not do anythng in uniaxial stress
            constraint=StressStrainConstraint.PLANE_STRAIN,
        )

        def left_boundary(x):
            return np.isclose(x[0], 0.0)
            # return np.isclose(x[0], -self.L/2.)

        def right_boundary(x):
            return np.isclose(x[0], self.L / 2.0)

        def boundary_top(x):
            return np.isclose(x[1], 1.0)

        def boundary_bottom(x):
            return np.isclose(x[1], 0.0)

        eps_mesh = (L / 2 / n_elements) * 1e-4

        def notch_marker(x):
            return x[0] <= self.W / 2.0 + eps_mesh
            # return (x[0]>=-self.W/2.0-eps_mesh) & (x[0]<=self.W/2.0+eps_mesh)

        def rest_marker(x):
            return x[0] >= self.W / 2.0 - eps_mesh
            # return (x[0]<=-self.W/2.0+eps_mesh) | (x[0]>=self.W/2.0-eps_mesh)

        cells_notch = df.mesh.locate_entities(mesh, mesh.topology.dim, notch_marker)
        cells_rest = df.mesh.locate_entities(mesh, mesh.topology.dim, rest_marker)
        print(cells_rest.shape, cells_notch.shape)
        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts

        assert cells_rest.size + cells_notch.size == num_cells

        displacement_left = df.fem.Constant(mesh, np.float64(0.0))
        displacement_right = df.fem.Constant(mesh, np.float64(self.deltaL / 2.0))
        displacement_top_bottom = df.fem.Constant(mesh, np.float64(0.0))

        entities_left = df.mesh.locate_entities_boundary(
            mixed_space.mesh, 1, left_boundary
        )
        entities_right = df.mesh.locate_entities_boundary(
            mixed_space.mesh, 1, right_boundary
        )
        entities_top = df.mesh.locate_entities_boundary(
            mixed_space.mesh, 1, boundary_top
        )
        entities_bottom = df.mesh.locate_entities_boundary(
            mixed_space.mesh, 1, boundary_bottom
        )

        dofs_left = df.fem.locate_dofs_topological(
            mixed_space.sub(0).sub(0), 1, entities_left
        )
        dofs_right = df.fem.locate_dofs_topological(
            mixed_space.sub(0).sub(0), 1, entities_right
        )
        dofs_top = df.fem.locate_dofs_topological(
            mixed_space.sub(0).sub(1), 1, entities_top
        )
        dofs_bottom = df.fem.locate_dofs_topological(
            mixed_space.sub(0).sub(1), 1, entities_bottom
        )

        bc_left = df.fem.dirichletbc(
            displacement_left, dofs_left, mixed_space.sub(0).sub(0)
        )
        bc_right = df.fem.dirichletbc(
            displacement_right, dofs_right, mixed_space.sub(0).sub(0)
        )
        bc_top = df.fem.dirichletbc(
            displacement_top_bottom, dofs_top, mixed_space.sub(0).sub(1)
        )
        bc_bottom = df.fem.dirichletbc(
            displacement_top_bottom, dofs_bottom, mixed_space.sub(0).sub(1)
        )

        problem = IncrSmallStrainGradientProblem(
            [(law, cells_rest), (law_notch, cells_notch)],
            u,
            [bc_left, bc_right, bc_top, bc_bottom],
            2,
            self.l,
        )

        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        solver.rtol = 1e-8
        solver.atol = 1e-8
        # solver.maxit = 200
        # solver.criterion="incremental"
        displacements = np.linspace(0.0, displacement_right.value, 20)
        for d in displacements:
            displacement_right.value = d
            n, converged = solver.solve(u)
            print("converged in ", n, "iterations. displ:", displacement_right.value)
            problem.update()

        self.problem = problem

class PeerlingsNumeric3D:
    def __init__(
        self,
        L: float,
        W_half_percentage: int,
        deltaL: float,
        alpha: float,
        E: float,
        eps_0: float,
        l: float,
        h_refinement: int,
    ):
        assert W_half_percentage < 50
        W = W_half_percentage / 100.0 * 2 * L
        self.L, self.W, self.deltaL, self.alpha = (
            L,
            W,
            deltaL,
            alpha,
        )  # 100.0, 10.0, 0.05, 0.1
        self.E, self.kappa0, self.l = E, eps_0, l  # 20000.0, 1.0e-4, 1.0
        n_elements = 50 * h_refinement
        mesh = df.mesh.create_box(
            MPI.COMM_WORLD,
            np.array([[0.0, 0.0, 0.0], [L / 2.0, 1.0, 1.0]]),
            [n_elements, 1,1],
            cell_type=df.mesh.CellType.hexahedron,
        )

        element_d = basix.ufl.element(
            ElementFamily.P, basix.CellType.hexahedron, 1, shape=(3,)
        )
        element_eps = basix.ufl.element(
            ElementFamily.P, basix.CellType.hexahedron, 1
        )
        mixed_element = basix.ufl.mixed_element([element_d, element_eps])
        mixed_space = df.fem.functionspace(mesh, mixed_element)
        u = df.fem.Function(mixed_space)

        law = PeerlingsGradientPerfectDamage(
            parameters={
                "E": self.E,
                "nu": 0.3,
                "eps_0": self.kappa0,
                "omega_max": 1.0,
            },  # nu does not do anythng in uniaxial stress
            constraint=StressStrainConstraint.FULL,
        )

        law_notch = PeerlingsGradientPerfectDamage(
            parameters={
                "E": self.E * (1.0 - self.alpha),
                "nu": 0.3,
                "eps_0": self.kappa0,
                "omega_max": 1.0,
            },  # nu does not do anythng in uniaxial stress
            constraint=StressStrainConstraint.FULL,
        )

        def left_boundary(x):
            return np.isclose(x[0], 0.0)
            # return np.isclose(x[0], -self.L/2.)

        def right_boundary(x):
            return np.isclose(x[0], self.L / 2.0)

        def boundary_top_bottom(x):
            return np.isclose(x[1], 1.0) | np.isclose(x[1], 0.0)
        
        def boundary_sides(x):
            return np.isclose(x[2], 0.0) | np.isclose(x[2], 1.0)

        eps_mesh = (L / 2 / n_elements) * 1e-4

        def notch_marker(x):
            return x[0] <= self.W / 2.0 + eps_mesh
            # return (x[0]>=-self.W/2.0-eps_mesh) & (x[0]<=self.W/2.0+eps_mesh)

        def rest_marker(x):
            return x[0] >= self.W / 2.0 - eps_mesh
            # return (x[0]<=-self.W/2.0+eps_mesh) | (x[0]>=self.W/2.0-eps_mesh)

        cells_notch = df.mesh.locate_entities(mesh, mesh.topology.dim, notch_marker)
        cells_rest = df.mesh.locate_entities(mesh, mesh.topology.dim, rest_marker)
        print(cells_rest.shape, cells_notch.shape)
        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts

        assert cells_rest.size + cells_notch.size == num_cells

        displacement_zero = df.fem.Constant(mesh, np.float64(0.0))
        displacement_right = df.fem.Constant(mesh, np.float64(self.deltaL / 2.0))

        entities_left = df.mesh.locate_entities_boundary(
            mixed_space.mesh, 2, left_boundary
        )
        entities_right = df.mesh.locate_entities_boundary(
            mixed_space.mesh, 2, right_boundary
        )
        entities_top_bottom = df.mesh.locate_entities_boundary(
            mixed_space.mesh, 2, boundary_top_bottom
        )
        entities_sides = df.mesh.locate_entities_boundary(
            mixed_space.mesh, 2, boundary_sides
        )

        dofs_left = df.fem.locate_dofs_topological(
            mixed_space.sub(0).sub(0), 2, entities_left
        )
        dofs_right = df.fem.locate_dofs_topological(
            mixed_space.sub(0).sub(0), 2, entities_right
        )
        dofs_top_bottom = df.fem.locate_dofs_topological(
            mixed_space.sub(0).sub(1), 2, entities_top_bottom
        )
        dofs_sides = df.fem.locate_dofs_topological(
            mixed_space.sub(0).sub(2), 2, entities_sides
        )

        bc_left = df.fem.dirichletbc(
            displacement_zero, dofs_left, mixed_space.sub(0).sub(0)
        )
        bc_right = df.fem.dirichletbc(
            displacement_right, dofs_right, mixed_space.sub(0).sub(0)
        )
        bc_top_bottom = df.fem.dirichletbc(
            displacement_zero, dofs_top_bottom, mixed_space.sub(0).sub(1)
        )
        bc_sides = df.fem.dirichletbc(
            displacement_zero, dofs_sides, mixed_space.sub(0).sub(2)
        )

        problem = IncrSmallStrainGradientProblem(
            [(law, cells_rest), (law_notch, cells_notch)],
            u,
            [bc_left, bc_right, bc_top_bottom, bc_sides],
            2,
            self.l,
        )

        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        solver.rtol = 1e-8
        solver.atol = 1e-8
        # solver.maxit = 200
        # solver.criterion="incremental"
        displacements = np.linspace(0.0, displacement_right.value, 20)
        for d in displacements:
            displacement_right.value = d
            n, converged = solver.solve(u)
            print("converged in ", n, "iterations. displ:", displacement_right.value)
            problem.update()

        self.problem = problem


def test_peerlings_1d():
    analytic = PeerlingsAnalytic(100.0, 5, 0.05, 0.1, 20000.0, 1e-4, 1.0)
    numeric = PeerlingsNumeric(100.0, 5, 0.05, 0.1, 20000.0, 1e-4, 1.0, 2)
    numeric2D = PeerlingsNumericPlaneStrain(100.0, 5, 0.05, 0.1, 20000.0, 1e-4, 1.0, 2)
    numeric3D = PeerlingsNumeric3D(100.0, 5, 0.05, 0.1, 20000.0, 1e-4, 1.0, 2)

    u, eps_nonlocal = numeric.problem.solution.current.split()
    u = u.collapse()
    eps_nonlocal = eps_nonlocal.collapse()
    x_nodes = eps_nonlocal.function_space.tabulate_dof_coordinates()[:, 0].flatten()
    
    e_exact = [analytic.e(x) for x in x_nodes]
    
    u_2d, eps_nonlocal_2d = numeric2D.problem.solution.current.split()
    eps_nonlocal_2d = eps_nonlocal_2d.collapse()
    x_nodes_2d = eps_nonlocal_2d.function_space.tabulate_dof_coordinates()[:, 0].flatten()
    e_exact_2d = [analytic.e(x) for x in x_nodes_2d]
    
    u_3d, eps_nonlocal_3d = numeric3D.problem.solution.current.split()
    eps_nonlocal_3d = eps_nonlocal_3d.collapse()
    x_nodes_3d = eps_nonlocal_3d.function_space.tabulate_dof_coordinates()[:, 0].flatten()
    e_exact_3d = [analytic.e(x) for x in x_nodes_3d]

    assert (
        np.linalg.norm(eps_nonlocal.x.array - e_exact) / np.linalg.norm(e_exact)
    ) < 1e-2

    abs_error_2d = np.linalg.norm(eps_nonlocal_2d.x.array - e_exact_2d)
    rel_error_2d =  abs_error_2d/ np.linalg.norm(e_exact_2d)
    assert (
        rel_error_2d
    ) < 1e-2, f"absolute error {abs_error_2d}, relative error {rel_error_2d}"
    
    abs_error_3d = np.linalg.norm(eps_nonlocal_3d.x.array - e_exact_3d)
    rel_error_3d =  abs_error_3d/ np.linalg.norm(e_exact_3d)
    assert (
        rel_error_3d
    ) < 1e-2, f"absolute error {abs_error_3d}, relative error {rel_error_3d}"
