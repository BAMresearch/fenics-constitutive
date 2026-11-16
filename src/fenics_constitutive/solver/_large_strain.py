from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import basix.ufl
import dolfinx as df
import numpy as np
import ufl
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc

from fenics_constitutive.models.interfaces import LargeStrainModel

from .utils import ufl_mandel, ufl_tensor_from_mandel

__all__ = ["ConstitutiveError", "LargeStrainProblem", "solve_incrementally"]

class ConstitutiveError(RuntimeError):
    """A constitutive model could not be evaluated. Raised on all MPI ranks together."""


@dataclass
class _LawOnCells:
    """A law together with the quadrature points of its cells and its history."""

    law: LargeStrainModel
    points: np.ndarray
    history_0: dict[str, np.ndarray] | None
    history_1: dict[str, np.ndarray] | None
    tau: np.ndarray = field(init=False)
    tangent: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.tau = np.zeros(6 * self.points.size)
        self.tangent = np.zeros(36 * self.points.size)

    @staticmethod
    def create(
        law: LargeStrainModel, cells: np.ndarray, quadrature_space: df.fem.FunctionSpace
    ) -> _LawOnCells:
        points = quadrature_space.dofmap.list[cells].flatten()
        history_0 = None
        history_1 = None
        initial = law.initial_history()
        if law.history_dim is not None:
            if initial is None:
                msg = "A law with history must define its initial history."
                raise ValueError(msg)
            history_0 = {
                key: np.tile(np.asarray(initial[key], dtype=np.float64).flatten(), points.size)
                for key in law.history_dim
            }
            history_1 = {key: value.copy() for key, value in history_0.items()}
        return _LawOnCells(law, points, history_0, history_1)

    def evaluate(
        self,
        t: float,
        del_t: float,
        F: df.fem.Function,
        tau: df.fem.Function,
        tangent: df.fem.Function | None,
    ) -> None:
        if self.history_0 is not None and self.history_1 is not None:
            for key, value in self.history_0.items():
                self.history_1[key][:] = value
        F_local = F.x.array.reshape(-1, 9)[self.points].flatten()
        self.law.evaluate(
            t,
            del_t,
            F_local,
            self.tau,
            self.tangent if tangent is not None else None,
            self.history_1,
        )
        tau.x.array.reshape(-1, 6)[self.points] = self.tau.reshape(-1, 6)
        if tangent is not None:
            tangent.x.array.reshape(-1, 36)[self.points] = self.tangent.reshape(-1, 36)

    def update(self) -> None:
        if self.history_0 is not None and self.history_1 is not None:
            for key, value in self.history_1.items():
                self.history_0[key][:] = value


class LargeStrainProblem(NonlinearProblem):
    r"""
    A large-strain problem for models following `LargeStrainModel`, formulated with the
    Kirchhoff stress. To be used with the dolfinx `NewtonSolver`.

    The unknown is the total displacement $\mathbf u$ on the reference configuration; the mesh
    is never moved. With $\mathbf g_v = \nabla\mathbf v\,\mathbf F^{-1}$, the residual is

    $$R = \int_{\Omega_0}\boldsymbol\tau:\mathbf g_v\,\mathrm dX - R_{ext}$$

    This is the weak form of $\mathrm{div}_x\boldsymbol\sigma + \mathbf b = \mathbf 0$ in the current
    configuration, $\int_\Omega\boldsymbol\sigma:\nabla_x\mathbf v\,\mathrm dx$, pulled back to the
    reference configuration with $\nabla_x\mathbf v = \mathbf g_v$ and $\mathrm dx = J\,\mathrm dX$.
    It equals $\int_{\Omega_0}\mathbf P:\nabla\mathbf v\,\mathrm dX$. With
    $\mathbf l = \nabla\delta\mathbf u\,\mathbf F^{-1} = \mathbf d + \mathbf w$, its linearization is

    $$\delta R = \int_{\Omega_0}(\mathbb a:\mathbf d):\mathbf g_v
        + (\mathbf w\boldsymbol\tau - \boldsymbol\tau\mathbf w):\mathbf g_v
        - \boldsymbol\tau:(\mathbf g_v\mathbf l)\,\mathrm dX .$$

    Boundary conditions and loads:

    - Dirichlet conditions prescribe the total displacement $\mathbf u = \mathbf x - \mathbf X$ at
      reference positions $\mathbf X$, e.g. a superposed rotation $\mathbf R$ of the point
      $\mathbf X$ is $\mathbf u = (\mathbf R - \mathbf I)\mathbf X$.
    - `external_forces` are integrated over the reference mesh (`ufl.dx`, `ufl.ds` act on
      $\Omega_0$, $\partial\Omega_0$, and `ufl.FacetNormal` is $\mathbf N$), so they must be
      nominal quantities. A body force $\mathbf b$ per current volume becomes
      $\int_{\Omega_0}J\,\mathbf b\cdot\mathbf v\,\mathrm dX$; a dead nominal traction $\mathbf t_0$
      (force per reference area) is simply $\int_{\partial\Omega_0}\mathbf t_0\cdot\mathbf v\,\mathrm dA$.
    - Loads that depend on the deformation, e.g. a pressure $p$ on the current surface, must be
      pulled back with Nanson's formula $\mathbf n\,\mathrm da = J\mathbf F^{-T}\mathbf N\,\mathrm dA$:
      $-\int_{\partial\Omega_0}p\,J\mathbf F^{-T}\mathbf N\cdot\mathbf v\,\mathrm dA$ with
      $\mathbf F = \mathbf I + \nabla\mathbf u$ written as a ufl expression of `u`. Such loads are
      only included in the residual; their contribution to the Jacobian is missing, which slows
      down the Newton convergence but does not change the solution.

    Args:
        laws: A single law for the whole domain or a list of tuples of a law and the local
            cell indices it is used on.
        u: The total displacement field. This is the unknown in the nonlinear problem.
        bcs: The Dirichlet boundary conditions on the total displacement.
        q_degree: The quadrature degree.
        del_t: The time increment that is passed to the laws.
        external_forces: ufl forms (linear in the test function) which are subtracted from the
            residual, formulated on the reference configuration, see above.
        form_compiler_options: The options for the form compiler.
        jit_options: The options for the JIT compiler.
    """

    def __init__(
        self,
        laws: list[tuple[LargeStrainModel, np.ndarray]] | LargeStrainModel,
        u: df.fem.Function,
        bcs: list[df.fem.DirichletBC],
        q_degree: int,
        del_t: float = 1.0,
        external_forces: list[ufl.Form] | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
    ) -> None:
        mesh = u.function_space.mesh
        if mesh.geometry.dim != 3:
            msg = "LargeStrainProblem is only implemented for 3D meshes."
            raise NotImplementedError(msg)
        cell_map = mesh.topology.index_map(mesh.topology.dim)
        if isinstance(laws, LargeStrainModel):
            laws = [(laws, np.arange(cell_map.size_local, dtype=np.int32))]

        def quadrature_space(shape: tuple[int, ...]) -> df.fem.FunctionSpace:
            element = basix.ufl.quadrature_element(
                mesh.topology.cell_name(), value_shape=shape, degree=q_degree
            )
            return df.fem.functionspace(mesh, element)

        # `F` is a method of NonlinearProblem, hence the long name
        self.deformation_gradient = df.fem.Function(quadrature_space((3, 3)), name="F")
        self.tau = df.fem.Function(quadrature_space((6,)), name="tau")
        self.tangent = df.fem.Function(quadrature_space((6, 6)), name="dtau_dd")

        # Only owned cells are evaluated; tau and the tangent of ghost cells are scattered.
        def owned(cells: np.ndarray) -> np.ndarray:
            cells = np.asarray(cells, dtype=np.int32)
            return cells[cells < cell_map.size_local]

        self._laws = [
            _LawOnCells.create(law, owned(cells), self.deformation_gradient.function_space)
            for law, cells in laws
        ]

        self.u = u
        self.u_previous = u.copy()
        self.time = 0.0
        self.del_t = del_t
        self._comm = mesh.comm

        self._F_expr = df.fem.Expression(
            ufl.Identity(3) + ufl.grad(u),
            self.deformation_gradient.function_space.element.interpolation_points(),
        )

        v = ufl.TestFunction(u.function_space)
        du = ufl.TrialFunction(u.function_space)
        self.dx = ufl.dx(
            metadata={"quadrature_degree": q_degree, "quadrature_scheme": "default"}
        )
        F_inv = ufl.inv(self.deformation_gradient)
        g_v = ufl.grad(v) * F_inv
        l = ufl.grad(du) * F_inv
        w = ufl.skew(l)
        tau = ufl_tensor_from_mandel(self.tau)

        residual = ufl.inner(self.tau, ufl_mandel(g_v)) * self.dx
        if external_forces is not None:
            residual -= sum(external_forces)
        jacobian = (
            ufl.inner(ufl.dot(self.tangent, ufl_mandel(l)), ufl_mandel(g_v))
            + ufl.inner(w * tau - tau * w, g_v)
            - ufl.inner(tau, g_v * l)
        ) * self.dx

        super().__init__(
            residual,
            u,
            bcs=bcs,
            J=jacobian,
            form_compiler_options=form_compiler_options if form_compiler_options is not None else {},
            jit_options=jit_options if jit_options is not None else {},
        )

    @df.common.timed("large-strain-constitutive-evaluation")
    def form(self, x: PETSc.Vec) -> None:
        """Called by the Newton solver before the residual and Jacobian are assembled. Evaluates
        the laws at the current solution `x`.

        Raises:
            ConstitutiveError: On all ranks, if a law fails on any rank.
        """
        super().form(x)
        if x.handle != self.u.x.petsc_vec.handle:
            x.copy(self.u.x.petsc_vec)
            self.u.x.scatter_forward()
        self.evaluate(evaluate_tangent=True)

    def evaluate(self, evaluate_tangent: bool = True) -> None:
        """Evaluates the laws at the current displacement `u` and writes `tau` and the tangent.
        The committed history is not changed.

        Raises:
            ConstitutiveError: On all ranks, if a law fails on any rank.
        """
        self.deformation_gradient.interpolate(self._F_expr)
        self.deformation_gradient.x.scatter_forward()
        tangent = self.tangent if evaluate_tangent else None
        error = None
        for law in self._laws:
            try:
                law.evaluate(self.time, self.del_t, self.deformation_gradient, self.tau, tangent)
            except ValueError as e:
                error = str(e)
                break
        errors = self._comm.allgather(error)
        if any(e is not None for e in errors):
            msg = "; ".join(
                f"rank {rank}: {e}" for rank, e in enumerate(errors) if e is not None
            )
            raise ConstitutiveError(msg)
        self.tau.x.scatter_forward()
        if tangent is not None:
            tangent.x.scatter_forward()

    def update(self) -> None:
        """Commits the converged state: history, displacement and time."""
        for law in self._laws:
            law.update()
        self.u_previous.x.array[:] = self.u.x.array
        self.time += self.del_t

    def reset(self) -> None:
        """Restores the displacement of the last committed state, e.g. to cut a step."""
        self.u.x.array[:] = self.u_previous.x.array
        self.u.x.scatter_forward()

    def history(self, law_index: int = 0) -> dict[str, np.ndarray] | None:
        """The committed history of one law, one row per quadrature point."""
        history = self._laws[law_index].history_0
        if history is None:
            return None
        n_points = self._laws[law_index].points.size
        return {key: value.reshape(n_points, -1) for key, value in history.items()}

    def cauchy_stress(self) -> ufl.core.expr.Expr:
        r"""The Cauchy stress $\boldsymbol\sigma = \boldsymbol\tau/\det\mathbf F$ as a 3x3 ufl
        expression of the quadrature functions."""
        return ufl_tensor_from_mandel(self.tau) / ufl.det(self.deformation_gradient)

    def piola_stress(self) -> ufl.core.expr.Expr:
        r"""The first Piola-Kirchhoff stress $\mathbf P = \boldsymbol\tau\mathbf F^{-T}$ as a ufl
        expression of the quadrature functions."""
        return ufl_tensor_from_mandel(self.tau) * ufl.inv(self.deformation_gradient).T


def solve_incrementally(
    problem: LargeStrainProblem,
    solver: df.nls.petsc.NewtonSolver,
    apply_load: Callable[[float], None],
    n_steps: int,
    max_cuts: int = 8,
    callback: Callable[[float], None] | None = None,
) -> list[int]:
    """Applies a load from 0 to 1 in `n_steps` equal steps and halves the step whenever the
    Newton solver or a law fails.

    Args:
        problem: The problem.
        solver: The Newton solver of `problem`. It must raise on non-convergence (the default).
        apply_load: Sets the boundary conditions/loads for a load factor in (0, 1].
        n_steps: The number of steps without cutting.
        max_cuts: The maximal number of consecutive step halvings.
        callback: Called with the load factor after each converged and committed step.

    Returns:
        The number of Newton iterations of each converged step.
    """
    factor = 0.0
    step = 1.0 / n_steps
    cuts = 0
    iterations = []
    while factor < 1.0 - 1e-12:
        step = min(step, 1.0 - factor)
        apply_load(factor + step)
        try:
            n_iterations, converged = solver.solve(problem.u)
        except (ConstitutiveError, RuntimeError):
            converged = False
        if converged:
            problem.update()
            factor += step
            iterations.append(n_iterations)
            cuts = 0
            if callback is not None:
                callback(factor)
        else:
            problem.reset()
            cuts += 1
            if cuts > max_cuts:
                msg = f"No convergence at load factor {factor} after {max_cuts} step cuts."
                raise ConstitutiveError(msg)
            step /= 2.0
    return iterations
