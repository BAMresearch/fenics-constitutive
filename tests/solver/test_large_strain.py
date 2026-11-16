from __future__ import annotations

import dolfinx as df
import numpy as np
import pytest
import ufl
from dolfinx.nls.petsc import NewtonSolver
import dolfinx.fem.petsc
from mpi4py import MPI
from petsc4py import PETSc

from fenics_constitutive.models import (
    HenckyDruckerPragerHyperbolic3D,
    HenckyElasticity3D,
    HenckyMises3D,
    LargeStrainModel,
)
from fenics_constitutive.models.rust_models import IsotropicMises3D
from fenics_constitutive.models.utils import mandel_to_tensor
from fenics_constitutive.solver import (
    ConstitutiveError,
    IncrSmallStrainProblem,
    LargeStrainProblem,
    solve_incrementally,
)

MU, KAPPA, Y_0, H = 80.77, 166.67, 0.2, 10.0
E = 9.0 * KAPPA * MU / (3.0 * KAPPA + MU)
MISES = {
    "mu": np.array([MU]),
    "kappa": np.array([KAPPA]),
    "y_0": np.array([Y_0]),
    "h": np.array([H]),
}


def newton_solver(problem: LargeStrainProblem) -> NewtonSolver:
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.atol = 1e-11
    solver.rtol = 1e-11
    solver.max_it = 15
    return solver


def uniaxial_setup(law: LargeStrainModel, n: int = 2):
    """Unit cube with rollers on x=0, y=0, z=0 and a prescribed u_x on x=1."""
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
    V = df.fem.functionspace(mesh, ("Lagrange", 1, (3,)))
    u = df.fem.Function(V)
    fdim = mesh.topology.dim - 1
    stretch = df.fem.Constant(mesh, 0.0)
    zero = df.fem.Constant(mesh, 0.0)
    bcs = []
    for axis in range(3):
        facets = df.mesh.locate_entities_boundary(
            mesh, fdim, lambda x, a=axis: np.isclose(x[a], 0.0)
        )
        dofs = df.fem.locate_dofs_topological(V.sub(axis), fdim, facets)
        bcs.append(df.fem.dirichletbc(zero, dofs, V.sub(axis)))
    facets = df.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], 1.0))
    dofs = df.fem.locate_dofs_topological(V.sub(0), fdim, facets)
    bcs.append(df.fem.dirichletbc(stretch, dofs, V.sub(0)))
    problem = LargeStrainProblem(law, u, bcs, q_degree=1)
    return problem, stretch


def owned_points(problem: LargeStrainProblem, values: df.fem.Function) -> np.ndarray:
    bs = values.function_space.dofmap.index_map_bs
    n_owned = values.function_space.dofmap.index_map.size_local
    return values.x.array[: n_owned * bs].reshape(n_owned, -1)


def analytic_tau_11(lam: float, yield_stress: str = "kirchhoff") -> float:
    """Uniaxial Kirchhoff stress for Hencky-Mises with linear hardening under monotonic
    plastic loading, alpha = |ln lam| - |tau|/E. Kirchhoff yield: |tau| = y_0 + h alpha.
    Cauchy yield: |tau| / J = y_0 + h alpha with J = exp(tau / (3 kappa)) (isochoric flow)."""
    log_lam = np.log(lam)
    sign = np.sign(log_lam)
    tau = sign * (Y_0 + H * abs(log_lam)) / (1.0 + H / E)
    if yield_stress == "kirchhoff":
        return tau
    for _ in range(50):
        J = np.exp(tau / (3.0 * KAPPA))
        yield_value = Y_0 + H * (abs(log_lam) - abs(tau) / E)
        residual = tau - sign * J * yield_value
        tau -= residual / (1.0 - sign * J / (3.0 * KAPPA) * yield_value + J * H / E)
    return tau


@pytest.mark.parametrize("yield_stress", ["kirchhoff", "cauchy"])
@pytest.mark.parametrize("final_stretch", [1.5, 0.5], ids=["tension", "compression"])
def test_uniaxial_mises_analytic(final_stretch, yield_stress):
    problem, stretch = uniaxial_setup(HenckyMises3D(MISES, yield_stress=yield_stress))
    solver = newton_solver(problem)
    n_steps = 20
    for factor in np.linspace(0.0, 1.0, n_steps + 1)[1:]:
        lam = 1.0 + factor * (final_stretch - 1.0)
        stretch.value = lam - 1.0
        n_iterations, converged = solver.solve(problem.u)
        assert converged
        assert n_iterations <= 6
        problem.update()
        tau = owned_points(problem, problem.tau)
        tau_11 = tau[:, 0]
        if abs(np.log(lam)) * E > 1.5 * Y_0:
            assert np.allclose(tau_11, analytic_tau_11(lam, yield_stress), rtol=1e-8, atol=0.0)
            # the yield condition holds in the chosen stress measure
            J = np.linalg.det(owned_points(problem, problem.deformation_gradient).reshape(-1, 3, 3))
            alpha = problem.history()["history"][:, 6]
            scale = J if yield_stress == "cauchy" else 1.0
            assert np.allclose(np.abs(tau_11) / scale, Y_0 + H * alpha, rtol=1e-8, atol=0.0)
        # uniaxial stress state
        assert np.abs(tau[:, 1:]).max() <= 1e-9 * np.abs(tau[:, 0]).max()
    alpha = problem.history()["history"][:, 6]
    assert np.allclose(alpha, alpha[0]) and alpha[0] > 0.1


def homogeneous_setup(law: LargeStrainModel):
    """Unit cube with the displacement of a homogeneous deformation F on the whole boundary."""
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    V = df.fem.functionspace(mesh, ("Lagrange", 1, (3,)))
    u = df.fem.Function(V)
    u_bc = df.fem.Function(V)
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    facets = df.mesh.exterior_facet_indices(mesh.topology)
    bc = df.fem.dirichletbc(u_bc, df.fem.locate_dofs_topological(V, fdim, facets))
    problem = LargeStrainProblem(law, u, [bc], q_degree=1)

    def apply(F: np.ndarray) -> None:
        u_bc.interpolate(lambda x: (F - np.eye(3)) @ x)

    return problem, apply


def rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_superposed_rigid_rotation_rotates_cauchy_stress():
    problem, apply = homogeneous_setup(HenckyMises3D(MISES))
    solver = newton_solver(problem)
    F_stretch = np.diag([1.0, 1.0, 1.0])
    for lam in np.linspace(1.0, 1.3, 7)[1:]:
        F_stretch = np.diag([lam, lam**-0.5, lam**-0.5])
        apply(F_stretch)
        assert solver.solve(problem.u)[1]
        problem.update()

    def cauchy(F: np.ndarray) -> np.ndarray:
        return mandel_to_tensor(owned_points(problem, problem.tau)) / np.linalg.det(F)

    sigma_0 = cauchy(F_stretch)
    assert problem.history()["history"][0, 6] > 0.1
    for angle in np.linspace(0.0, np.pi / 2, 7)[1:]:
        R = rotation_z(angle)
        F = R @ F_stretch
        apply(F)
        assert solver.solve(problem.u)[1]
        problem.update()
        expected = R @ sigma_0 @ R.T
        assert np.allclose(cauchy(F), expected, rtol=0.0, atol=1e-8 * np.abs(sigma_0).max())


def shear_compression_block(make_problem, strain: float, n_steps: int = 5):
    mesh = df.mesh.create_box(
        MPI.COMM_WORLD, [np.zeros(3), np.ones(3)], [3, 3, 3], df.mesh.CellType.tetrahedron
    )
    V = df.fem.functionspace(mesh, ("Lagrange", 1, (3,)))
    u = df.fem.Function(V)
    fdim = mesh.topology.dim - 1
    bottom = df.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], 0.0))
    top = df.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], 1.0))
    top_value = df.fem.Constant(mesh, np.zeros(3))
    bcs = [
        df.fem.dirichletbc(np.zeros(3), df.fem.locate_dofs_topological(V, fdim, bottom), V),
        df.fem.dirichletbc(top_value, df.fem.locate_dofs_topological(V, fdim, top), V),
    ]
    problem = make_problem(mesh, u, bcs)
    solver = newton_solver(problem)
    for factor in np.linspace(0.0, 1.0, n_steps + 1)[1:]:
        top_value.value = factor * strain * np.array([1.0, -1.0, 0.0])
        assert solver.solve(u)[1]
        problem.update()
    return u, problem


def test_small_strain_limit():
    strain = 2e-3

    def large(mesh, u, bcs):
        return LargeStrainProblem(HenckyMises3D(MISES), u, bcs, q_degree=1)

    def small(mesh, u, bcs):
        return IncrSmallStrainProblem(IsotropicMises3D(MISES), u, bcs, q_degree=1)

    u_large, problem_large = shear_compression_block(large, strain)
    u_small, _ = shear_compression_block(small, strain)
    assert problem_large.history()["history"][:, 6].max() > 0.0, "not plastic"
    # both meshes are created identically, so the dof layouts agree
    index_map = u_small.function_space.dofmap.index_map
    n_owned = index_map.size_local * u_small.function_space.dofmap.index_map_bs
    difference = u_large.x.array[:n_owned] - u_small.x.array[:n_owned]
    error = np.sqrt(
        MPI.COMM_WORLD.allreduce(np.sum(difference**2), op=MPI.SUM)
        / MPI.COMM_WORLD.allreduce(np.sum(u_small.x.array[:n_owned] ** 2), op=MPI.SUM)
    )
    assert error < strain


def test_several_laws_match_single_law():
    strain = 0.05

    def single(mesh, u, bcs):
        return LargeStrainProblem(HenckyMises3D(MISES), u, bcs, q_degree=1)

    def split(mesh, u, bcs):
        cell_map = mesh.topology.index_map(mesh.topology.dim)
        cells = np.arange(cell_map.size_local + cell_map.num_ghosts, dtype=np.int32)
        midpoints = df.mesh.compute_midpoints(mesh, mesh.topology.dim, cells)
        lower = midpoints[:, 1] < 0.5
        laws = [(HenckyMises3D(MISES), cells[lower]), (HenckyMises3D(MISES), cells[~lower])]
        return LargeStrainProblem(laws, u, bcs, q_degree=1)

    u_single, _ = shear_compression_block(single, strain)
    u_split, _ = shear_compression_block(split, strain)
    assert np.allclose(u_single.x.array, u_split.x.array, rtol=0.0, atol=1e-12)


class FailOnLargeIncrement(LargeStrainModel):
    """Raises if F_11 changes by more than `max_increment` within one step."""

    def __init__(self, model: LargeStrainModel, max_increment: float) -> None:
        self.model = model
        self.max_increment = max_increment

    @property
    def history_dim(self):
        return {**self.model.history_dim, "F11": 1}

    def initial_history(self):
        return {**self.model.initial_history(), "F11": np.ones(1)}

    def evaluate(self, t, del_t, F, tau, tangent, history):
        if np.abs(F[0::9] - history["F11"]).max() > self.max_increment:
            msg = "increment too large"
            raise ValueError(msg)
        inner = {key: history[key] for key in self.model.history_dim}
        self.model.evaluate(t, del_t, F, tau, tangent, inner)
        history["F11"][:] = F[0::9]


def test_step_cutting():
    problem, stretch = uniaxial_setup(FailOnLargeIncrement(HenckyMises3D(MISES), 0.06))
    solver = newton_solver(problem)
    iterations = solve_incrementally(
        problem, solver, lambda factor: setattr(stretch, "value", 0.5 * factor), n_steps=5
    )
    # every step of 0.1 has to be cut at least once
    assert len(iterations) >= 10
    tau = owned_points(problem, problem.tau)
    assert np.allclose(tau[:, 0], analytic_tau_11(1.5), rtol=1e-8)


class FailOnRankZero(LargeStrainModel):
    def __init__(self, model: LargeStrainModel) -> None:
        self.model = model

    @property
    def history_dim(self):
        return self.model.history_dim

    def initial_history(self):
        return self.model.initial_history()

    def evaluate(self, t, del_t, F, tau, tangent, history):
        if MPI.COMM_WORLD.rank == 0 and np.abs(F[0::9] - 1.0).max() > 0.01:
            msg = "failure on rank 0"
            raise ValueError(msg)
        self.model.evaluate(t, del_t, F, tau, tangent, history)


@pytest.mark.mpi
def test_local_failure_raises_on_all_ranks():
    problem, stretch = uniaxial_setup(FailOnRankZero(HenckyElasticity3D(MISES)), n=4)
    solver = newton_solver(problem)
    stretch.value = 0.1
    with pytest.raises(ConstitutiveError, match="rank 0"):
        solver.solve(problem.u)
    problem.reset()
    assert np.all(problem.u.x.array == 0.0)


def test_drucker_prager_hyperbolic_compression():
    params = {
        "mu": np.array([MU]),
        "kappa": np.array([KAPPA]),
        "a": np.array([0.15]),
        "b": np.array([0.05]),
        "d": np.array([0.1]),
        "h": np.array([5.0]),
        "b_flow": np.array([0.03]),
    }
    problem, stretch = uniaxial_setup(HenckyDruckerPragerHyperbolic3D(params))
    solver = newton_solver(problem)
    iterations = solve_incrementally(
        problem, solver, lambda factor: setattr(stretch, "value", -0.3 * factor), n_steps=15
    )
    assert max(iterations) <= 8
    assert problem.history()["history"][:, 6].min() > 0.0


@pytest.mark.parametrize("yield_stress", ["kirchhoff", "cauchy"])
def test_jacobian_finite_differences(yield_stress):
    """The assembled Jacobian (material and geometric terms) agrees with central differences
    of the assembled residual at an inhomogeneous, plastic state with shear."""

    def large(mesh, u, bcs):
        law = HenckyMises3D(MISES, yield_stress=yield_stress)
        return LargeStrainProblem(law, u, bcs, q_degree=1)

    u, problem = shear_compression_block(large, strain=0.05)
    assert problem.history()["history"][:, 6].max() > 0.0, "not plastic"

    V = u.function_space
    n_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    direction = df.fem.Function(V)
    rng = np.random.default_rng(MPI.COMM_WORLD.rank)
    direction.x.array[:n_owned] = rng.standard_normal(n_owned)
    direction.x.scatter_forward()
    # Load 5% further than the committed state: at the committed state itself the trial
    # stress lies exactly on the yield surface, where the incremental map has a kink.
    u_0 = 1.05 * u.x.array
    u.x.array[:] = u_0
    problem.evaluate(evaluate_tangent=True)
    A = df.fem.petsc.assemble_matrix(problem.a)
    A.assemble()
    jacobian_times_direction = A.createVecLeft()
    A.mult(direction.x.petsc_vec, jacobian_times_direction)

    def residual(scale: float) -> np.ndarray:
        u.x.array[:] = u_0 + scale * direction.x.array
        problem.evaluate(evaluate_tangent=False)
        b = df.fem.petsc.assemble_vector(problem.L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        return b.array.copy()

    step = 1e-6
    fd = (residual(step) - residual(-step)) / (2.0 * step)
    analytical = jacobian_times_direction.array
    error = np.sqrt(MPI.COMM_WORLD.allreduce(np.sum((analytical - fd) ** 2), op=MPI.SUM))
    norm = np.sqrt(MPI.COMM_WORLD.allreduce(np.sum(analytical**2), op=MPI.SUM))
    assert error <= 1e-6 * norm


def test_residual_matches_reference_configuration_form():
    """The Kirchhoff form int tau : (grad_X v F^-1) dX is the current-configuration weak form
    int sigma : grad_x v dx pulled back; it must equal int P : grad_X v dX."""

    def large(mesh, u, bcs):
        return LargeStrainProblem(HenckyMises3D(MISES), u, bcs, q_degree=1)

    u, problem = shear_compression_block(large, strain=0.2)
    assert problem.history()["history"][:, 6].max() > 0.0, "not plastic"
    problem.evaluate(evaluate_tangent=False)

    v = ufl.TestFunction(u.function_space)
    reference_form = df.fem.form(ufl.inner(problem.piola_stress(), ufl.grad(v)) * problem.dx)
    vectors = []
    for form in [problem.L, reference_form]:
        b = df.fem.petsc.assemble_vector(form)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        vectors.append(b.array.copy())
    error = np.sqrt(MPI.COMM_WORLD.allreduce(np.sum((vectors[0] - vectors[1]) ** 2), op=MPI.SUM))
    norm = np.sqrt(MPI.COMM_WORLD.allreduce(np.sum(vectors[1] ** 2), op=MPI.SUM))
    assert error <= 1e-12 * norm
