from __future__ import annotations

from dataclasses import dataclass

import dolfinx as df
import numpy as np
import ufl
from dolfinx.fem.function import Function
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc

from fenics_constitutive.constitutive.interfaces import IncrSmallStrainModel
from fenics_constitutive.solver._spaces import ElementSpaces
from fenics_constitutive.constitutive.stress_strain import ufl_mandel_strain

from fenics_constitutive.solver.typesafe import fn_for
from ._incrementalunknowns import IncrementalDisplacement, IncrementalStress
from ._lawonsubmesh import LawOnSubMesh, create_law_on_submesh


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
            that the domain is homogenous. The cell indices should be local indices of the MPI-process.
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
    ) -> None:
        mesh = u.function_space.mesh
        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        if isinstance(laws, IncrSmallStrainModel):
            laws = [(laws, np.arange(0, num_cells, dtype=np.int32))]

        constraint = laws[0][0].constraint
        assert all(law[0].constraint == constraint for law in laws), (
            "All laws must have the same constraint"
        )

        element_spaces = ElementSpaces.create(mesh, constraint, q_degree)
        self.stress = IncrementalStress(element_spaces.stress_vector_space)
        self.tangent = fn_for(element_spaces.stress_tensor_space(mesh))

        self._law_on_submeshs: list[LawOnSubMesh] = []
        self.sim_time = SimulationTime(dt=del_t)

        self._law_on_submeshs = [
            create_law_on_submesh(law, local_cells, element_spaces)
            for law, local_cells in laws
        ]

        u_, du = ufl.TestFunction(u.function_space), ufl.TrialFunction(u.function_space)

        self.metadata = {"quadrature_degree": q_degree, "quadrature_scheme": "default"}
        self.dxm = ufl.dx(metadata=self.metadata)

        self.R_form = (
            ufl.inner(ufl_mandel_strain(u_, constraint), self.stress.current) * self.dxm
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
    def a(self) -> df.fem.Form:
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

        for law in self._law_on_submeshs:
            law.evaluate(self.sim_time, self.incr_disp, self.stress, self.tangent)

        self.stress.scatter_current()
        self.tangent.x.scatter_forward()

    def update(self) -> None:
        """
        Update the current displacement, stress and history.
        """
        self.incr_disp.update_previous()
        self.stress.update_previous()

        for law in self._law_on_submeshs:
            law.update_history()

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

        return [_history_or_none(law) for law in self._law_on_submeshs]

    @property
    def _history_1(self) -> list[dict[str, Function] | None]:
        """Return a list of history_1 dicts for all laws (for backward compatibility)."""

        def _history_or_none(law) -> dict[str, Function] | None:
            return law.history.history_1 if law.history else None

        return [_history_or_none(law) for law in self._law_on_submeshs]

    @property
    def _del_grad_u(self) -> list[Function]:
        """Return a list of inc_disp_grad Functions for all laws (for backward compatibility)."""
        return [law.displacement_gradient_fn for law in self._law_on_submeshs]
