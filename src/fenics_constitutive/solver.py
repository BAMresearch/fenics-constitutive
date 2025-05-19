from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import basix
import basix.ufl
import dolfinx as df
import numpy as np
import ufl
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.fem.function import Function
from petsc4py import PETSc

from .history import History
from .interfaces import IncrSmallStrainModel
from .maps import SubSpaceMap, build_subspace_map
from .stress_strain import ufl_mandel_strain


class LawContext(ABC):
    law: IncrSmallStrainModel
    displacement_gradient: DisplacementGradientFunction
    history: History | None

    @abstractmethod
    def update_stress_and_tangent(
        self, solver: IncrSmallStrainProblem
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @abstractmethod
    def map_to_parent(self, solver: IncrSmallStrainProblem) -> None: ...

    def displacement_gradient_array(self) -> np.ndarray:
        return self.displacement_gradient.displacement_gradient()

    def scatter_displacement_gradient(self) -> None:
        self.displacement_gradient.scatter()

    def step(self, solver: IncrSmallStrainProblem) -> None:
        """Perform a full constitutive update for this law context."""
        solver.incr_disp.evaluate_incremental_gradient(self.displacement_gradient)
        stress_input, tangent_input = self.update_stress_and_tangent(solver)
        history_input = self.history.advance() if self.history is not None else None
        with df.common.Timer("constitutive-law-evaluation"):
            self.law.evaluate(
                solver.sim_time.current,
                solver.sim_time.dt,
                self.displacement_gradient.displacement_gradient_fn.x.array,
                stress_input,
                tangent_input,
                history_input,
            )
        self.map_to_parent(solver)


@dataclass
class SingleLawContext(LawContext):
    law: IncrSmallStrainModel
    displacement_gradient: DisplacementGradientFunction
    history: History | None = None

    def update_stress_and_tangent(
        self, solver: IncrSmallStrainProblem
    ) -> tuple[np.ndarray, np.ndarray]:
        solver.stress.update_current()
        return solver.stress.current_array(), solver.tangent.x.array

    def map_to_parent(self, solver: IncrSmallStrainProblem) -> None:
        # No mapping needed in single law case
        pass


@dataclass
class MultiLawContext(LawContext):
    law: IncrSmallStrainModel
    displacement_gradient: DisplacementGradientFunction
    stress: df.fem.Function
    tangent: df.fem.Function
    submesh_map: SubSpaceMap
    history: History | None = None

    def update_stress_and_tangent(
        self, solver: IncrSmallStrainProblem
    ) -> tuple[np.ndarray, np.ndarray]:
        self.submesh_map.map_to_child(solver.stress.previous, self.stress)
        return self.stress.x.array, self.tangent.x.array

    def map_to_parent(self, solver: IncrSmallStrainProblem) -> None:
        self.submesh_map.map_to_parent(self.stress, solver.stress.current)
        self.submesh_map.map_to_parent(self.tangent, solver.tangent)


@dataclass(slots=True)
class DisplacementGradientFunction:
    cells: np.ndarray
    displacement_gradient_fn: df.fem.Function

    def evaluate_expression(self, expression: df.fem.Expression) -> None:
        # expression.eval(self.cells, self.displacement_gradient())
        self.displacement_gradient_fn.interpolate(
            expression,
            cells0=self.cells,
            cells1=np.arange(self.cells.size, dtype=np.int32),
        )
        self.scatter()

    def displacement_gradient(self) -> np.ndarray:
        return self.displacement_gradient_fn.x.array.reshape(self.cells.size, -1)

    def scatter(self) -> None:
        self.displacement_gradient_fn.x.scatter_forward()


@dataclass
class IncrementalDisplacement:
    u: df.fem.Function
    q_degree: int

    def __post_init__(self) -> None:
        mesh = self.u.function_space.mesh
        basix_celltype = getattr(basix.CellType, mesh.topology.cell_type.name)
        q_points, _ = basix.make_quadrature(basix_celltype, self.q_degree)
        self.current = self.u
        self.previous = self.u.copy()
        self._expr = df.fem.Expression(
            ufl.nabla_grad(self.current - self.previous), q_points
        )

    def update_previous(self) -> None:
        self.previous.x.array[:] = self.current.x.array
        self.previous.x.scatter_forward()

    def update_current(self, x: np.ndarray) -> None:
        """Copy the solution vector x into the current displacement and update ghosts."""
        x.copy(self.current.x.petsc_vec)
        self.current.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

    def evaluate_incremental_gradient(
        self, displacement_gradient: DisplacementGradientFunction
    ) -> None:
        """Evaluate the incremental displacement gradient function"""
        displacement_gradient.evaluate_expression(self._expr)


class IncrementalStress:
    __slots__ = ("_current", "_previous")

    def __init__(self, function_space):
        self._current = fn_for(function_space)
        self._previous = fn_for(function_space)

    @property
    def current(self) -> df.fem.Function:
        return self._current

    @property
    def previous(self) -> df.fem.Function:
        return self._previous

    def current_array(self) -> np.ndarray:
        return self.current.x.array

    def update_previous(self) -> None:
        self._previous.x.array[:] = self._current.x.array
        self._previous.x.scatter_forward()

    def update_current(self) -> None:
        self._current.x.array[:] = self._previous.x.array
        self._current.x.scatter_forward()

    def scatter_current(self) -> None:
        self._current.x.scatter_forward()


@dataclass(slots=True)
class SimulationTime:
    dt: float
    current: float = 0

    def advance(self) -> None:
        self.current += self.dt


class IncrSmallStrainProblem(NonlinearProblem):
    """
    A nonlinear problem for incremental small strain models. To be used with
    the dolfinx NewtonSolver.

    Args:
        laws: A list of tuples where the first element is the constitutive law and the second
            element is the cells for the submesh. If only one law is provided, it is assumed
            that the domain is homogenous.
        u: The displacement field. This is the unknown in the nonlinear problem.
        bcs: The Dirichlet boundary conditions.
        q_degree: The quadrature degree (Polynomial degree which the quadrature rule needs to integrate exactly).
        del_t: The time increment.
        form_compiler_options: The options for the form compiler.
        jit_options: The options for the JIT compiler.

    Note:
        If `super().__init__(R, u, bcs, dR)` is called within the __init__ method,
        the user cannot add Neumann BCs. Therefore, the compilation (i.e. call to
        `super().__init__()`) is done when `df.nls.petsc.NewtonSolver` is initialized.
        The solver will call `self._A = fem.petsc.create_matrix(problem.a)` and hence
        we override the property ``a`` of NonlinearProblem to ensure that the form is compiled.
    """

    def __init__(
        self,
        laws: list[tuple[IncrSmallStrainModel, np.ndarray]] | IncrSmallStrainModel,
        u: df.fem.Function,
        bcs: list[df.fem.DirichletBC],
        q_degree: int,
        del_t: float = 1.0,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
    ):
        mesh = u.function_space.mesh
        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        cells = np.arange(0, num_cells, dtype=np.int32)
        if isinstance(laws, IncrSmallStrainModel):
            laws = [(laws, cells)]

        constraint = laws[0][0].constraint
        assert all(law[0].constraint == constraint for law in laws), (
            "All laws must have the same constraint"
        )

        gdim = mesh.geometry.dim
        assert constraint.geometric_dim == gdim, (
            "Geometric dimension mismatch between mesh and laws"
        )

        QVe = basix.ufl.quadrature_element(
            mesh.topology.cell_name(),
            value_shape=(constraint.stress_strain_dim,),
            degree=q_degree,
        )
        QTe = basix.ufl.quadrature_element(
            mesh.topology.cell_name(),
            value_shape=(
                constraint.stress_strain_dim,
                constraint.stress_strain_dim,
            ),
            degree=q_degree,
        )
        Q_grad_u_e = basix.ufl.quadrature_element(
            mesh.topology.cell_name(), value_shape=(gdim, gdim), degree=q_degree
        )
        QV = df.fem.functionspace(mesh, QVe)
        QT = df.fem.functionspace(mesh, QTe)

        self._law_contexts: list[LawContext] = []
        self.sim_time = SimulationTime(dt=del_t)

        if len(laws) == 1:
            # Single law case
            law, cells = laws[0]
            QT_subspace = df.fem.functionspace(mesh, QTe)
            tangent_fn: df.fem.Function = fn_for(QT_subspace)

            Q_grad_u_subspace = df.fem.functionspace(mesh, Q_grad_u_e)
            inc_disp_grad_fn = fn_for(Q_grad_u_subspace)
            disp_grad = DisplacementGradientFunction(cells, inc_disp_grad_fn)
            self._law_contexts.append(
                SingleLawContext(
                    law=law,
                    displacement_gradient=disp_grad,
                    history=History.try_create(law, mesh, q_degree),
                )
            )
        else:
            # Multi law case
            for law, cells in laws:
                subspace_map_tuple = build_subspace_map(cells, QV, return_subspace=True)
                if len(subspace_map_tuple) == 3:
                    subspace_map, submesh, QV_subspace = subspace_map_tuple
                else:
                    subspace_map, submesh = subspace_map_tuple
                    QV_subspace = QV  # fallback
                stress_fn = fn_for(QV_subspace)
                Q_grad_u_subspace = df.fem.functionspace(submesh, Q_grad_u_e)
                inc_disp_grad_fn = fn_for(Q_grad_u_subspace)
                QT_subspace = df.fem.functionspace(submesh, QTe)
                tangent_fn: df.fem.Function = fn_for(QT_subspace)
                disp_grad = DisplacementGradientFunction(cells, inc_disp_grad_fn)
                self._law_contexts.append(
                    MultiLawContext(
                        law=law,
                        displacement_gradient=disp_grad,
                        stress=stress_fn,
                        tangent=tangent_fn,
                        submesh_map=subspace_map,
                        history=History.try_create(law, submesh, q_degree),
                    )
                )

        self.stress = IncrementalStress(QV)
        self.tangent = fn_for(QT)

        u_, du = ufl.TestFunction(u.function_space), ufl.TrialFunction(u.function_space)

        self.metadata = {"quadrature_degree": q_degree, "quadrature_scheme": "default"}
        self.dxm = ufl.dx(metadata=self.metadata)

        self.R_form = (
            ufl.inner(ufl_mandel_strain(u_, constraint), self.stress_1) * self.dxm
        )
        self.dR_form = (
            ufl.inner(
                ufl_mandel_strain(du, constraint),
                ufl.dot(self.tangent, ufl_mandel_strain(u_, constraint)),
            )
            * self.dxm
        )

        self._bcs = bcs
        self._form_compiler_options = form_compiler_options
        self._jit_options = jit_options

        self.incr_disp = IncrementalDisplacement(u, q_degree)

    @property
    def a(self) -> df.fem.FormMetaClass:
        """Compiled bilinear form (the Jacobian form)"""

        if not hasattr(self, "_a"):
            # ensure compilation of UFL forms
            super().__init__(
                self.R_form,
                self.incr_disp.current,
                self._bcs,
                self.dR_form,
                form_compiler_options=self._form_compiler_options
                if self._form_compiler_options is not None
                else {},
                jit_options=self._jit_options if self._jit_options is not None else {},
            )

        return self._a

    @df.common.timed("constitutive-form-evaluation")
    def form(self, x: PETSc.Vec) -> None:
        """This function is called before the residual or Jacobian is
        computed. This is usually used to update ghost values, but here
        we use it to update the stress, tangent and history.

        Args:
            x: The vector containing the latest solution

        """
        super().form(x)
        self.incr_disp.update_current(x)

        for law_ctx in self._law_contexts:
            law_ctx.step(self)

        self.stress.scatter_current()
        self.tangent.x.scatter_forward()

    def update(self) -> None:
        """
        Update the current displacement, stress and history.
        """
        self.incr_disp.update_previous()
        self.stress.update_previous()

        for law in self._law_contexts:
            if law.history is not None:
                law.history.commit()

        self.sim_time.advance()

    # -------------------------------------------------------------------
    # NOTE: The following properties are used for backward compatibility
    # -------------------------------------------------------------------

    @property
    def _time(self) -> float:
        return self.sim_time.current

    @_time.setter
    def _time(self, value: float) -> None:
        self.sim_time.current = value

    @property
    def _del_t(self) -> float:
        return self.sim_time.dt

    @_del_t.setter
    def _del_t(self, value: float) -> None:
        self.sim_time.dt = value

    @property
    def _u(self) -> df.fem.Function:
        return self.incr_disp.current

    @property
    def _u0(self) -> df.fem.Function:
        return self.incr_disp.previous

    @property
    def stress_0(self) -> df.fem.Function:
        return self.stress.previous

    @property
    def stress_1(self) -> df.fem.Function:
        return self.stress.current

    @property
    def _history_0(self) -> list[dict[str, Function] | None]:
        """Return a list of history_0 dicts for all laws (for backward compatibility)."""

        def _history_or_none(law) -> dict[str, Function] | None:
            return law.history.history_0 if law.history else None

        return [_history_or_none(law) for law in self._law_contexts]

    @property
    def _history_1(self) -> list[dict[str, Function] | None]:
        """Return a list of history_1 dicts for all laws (for backward compatibility)."""

        def _history_or_none(law) -> dict[str, Function] | None:
            return law.history.history_1 if law.history else None

        return [_history_or_none(law) for law in self._law_contexts]

    @property
    def _del_grad_u(self) -> list[Function]:
        """Return a list of inc_disp_grad Functions for all laws (for backward compatibility)."""
        return [
            law.displacement_gradient.displacement_gradient_fn
            for law in self._law_contexts
        ]


def fn_for(space: df.fem.FunctionSpace) -> df.fem.Function:
    """Create a Function for the given FunctionSpace."""
    function = df.fem.Function(space)
    assert isinstance(function, df.fem.Function)
    return function
