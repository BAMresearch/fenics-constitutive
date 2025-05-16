from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

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


@dataclass
class LawData:
    law: IncrSmallStrainModel
    cells: np.ndarray
    del_grad_u: df.fem.Function
    stress: df.fem.Function | None  # None for single-law/homogeneous case
    tangent: df.fem.Function
    history: History | None = None


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

        self.submesh_maps: list[SubSpaceMap] = []
        self._laws: list[LawData] = []

        self._del_t = del_t  # time increment
        self._time = 0  # global time will be updated in the update method

        # if len(laws) > 1:
        for law, cells in laws:
            # default case for homogenous domain
            submesh = mesh
            stress_fn = None

            if len(laws) > 1:
                # ### submesh and subspace for strain, stress
                subspace_map_tuple = build_subspace_map(cells, QV, return_subspace=True)
                if len(subspace_map_tuple) == 3:
                    subspace_map, submesh, QV_subspace = subspace_map_tuple
                else:
                    subspace_map, submesh = subspace_map_tuple
                    QV_subspace = QV  # fallback
                self.submesh_maps.append(subspace_map)
                stress_fn = fn_for(QV_subspace)

            # subspace for grad u
            Q_grad_u_subspace = df.fem.functionspace(submesh, Q_grad_u_e)
            del_grad_u_fn = df.fem.Function(Q_grad_u_subspace)

            # subspace for tanget
            QT_subspace = df.fem.functionspace(submesh, QTe)
            tangent_fn = df.fem.Function(QT_subspace)

            self._laws.append(
                LawData(
                    law=law,
                    cells=cells,
                    del_grad_u=del_grad_u_fn,
                    stress=stress_fn,
                    history=History.try_create(law, submesh, q_degree),
                    tangent=tangent_fn,
                )
            )

        self.stress_0 = fn_for(QV)
        self.stress_1 = fn_for(QV)
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

        self._u = u
        self._u0 = u.copy()
        self._bcs = bcs
        self._form_compiler_options = form_compiler_options
        self._jit_options = jit_options

        basix_celltype = getattr(basix.CellType, mesh.topology.cell_type.name)
        self.q_points, _ = basix.make_quadrature(basix_celltype, q_degree)

        self.del_grad_u_expr = df.fem.Expression(
            ufl.nabla_grad(self._u - self._u0), self.q_points
        )

    @property
    def _history_0(self) -> list[Dict[str, Function] | None]:
        """Return a list of history_0 dicts for all laws (for backward compatibility)."""

        def _history_or_none(law_data: LawData) -> dict[str, Function] | None:
            return law_data.history.history_0 if law_data.history else None

        return [_history_or_none(law_data) for law_data in self._laws]

    @property
    def _history_1(self) -> list[Dict[str, Function] | None]:
        """Return a list of history_1 dicts for all laws (for backward compatibility)."""

        def _history_or_none(law_data: LawData) -> dict[str, Function] | None:
            return law_data.history.history_1 if law_data.history else None

        return [_history_or_none(law_data) for law_data in self._laws]

    @property
    def _del_grad_u(self) -> list[Function]:
        """Return a list of del_grad_u Functions for all laws (for backward compatibility)."""
        return [law_data.del_grad_u for law_data in self._laws]

    @property
    def a(self) -> df.fem.FormMetaClass:
        """Compiled bilinear form (the Jacobian form)"""

        if not hasattr(self, "_a"):
            # ensure compilation of UFL forms
            super().__init__(
                self.R_form,
                self._u,
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
        # this copies the data from the vector x to the function _u
        x.copy(self._u.x.petsc_vec)
        self._u.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        # This assertion can fail, even if everything is correct.
        # Left here, because I would like the check to work someday again.
        # assert (
        #    x.array.data == self._u.vector.array.data
        # ), f"The solution vector must be the same as the one passed to the MechanicsProblem. Got {x.array.data} and {self._u.vector.array.data}"

        # if len(self.laws) > 1:
        for k, law_data in enumerate(self._laws):
            law = law_data.law
            cells = law_data.cells

            law_data.del_grad_u.interpolate(
                self.del_grad_u_expr,
                cells0=cells,
                cells1=np.arange(cells.size, dtype=np.int32),
            )

            law_data.del_grad_u.x.scatter_forward()
            if len(self._laws) > 1 and law_data.stress is not None:
                self.submesh_maps[k].map_to_child(self.stress_0, law_data.stress)
                stress_input = law_data.stress.x.array
                tangent_input = law_data.tangent.x.array
            else:
                self.stress_1.x.array[:] = self.stress_0.x.array
                self.stress_1.x.scatter_forward()
                stress_input = self.stress_1.x.array
                tangent_input = self.tangent.x.array

            history_input = None
            if law_data.history is not None:
                history_input = law_data.history.advance()

            with df.common.Timer("constitutive-law-evaluation"):
                law.evaluate(
                    self._time,
                    self._del_t,
                    law_data.del_grad_u.x.array,
                    stress_input,
                    tangent_input,
                    history_input,
                )

            if len(self._laws) > 1 and law_data.stress is not None:
                self.submesh_maps[k].map_to_parent(law_data.stress, self.stress_1)
                self.submesh_maps[k].map_to_parent(law_data.tangent, self.tangent)

        self.stress_1.x.scatter_forward()
        self.tangent.x.scatter_forward()

    def update(self) -> None:
        """
        Update the current displacement, stress and history.
        """
        self._u0.x.array[:] = self._u.x.array
        self._u0.x.scatter_forward()

        self.stress_0.x.array[:] = self.stress_1.x.array
        self.stress_0.x.scatter_forward()

        for k, law_data in enumerate(self._laws):
            if law_data.history is not None:
                law_data.history.commit()

        # time update
        self._time += self._del_t


def fn_for(space: df.fem.FunctionSpace) -> df.fem.Function:
    """Create a Function for the given FunctionSpace."""
    function = df.fem.Function(space)
    assert isinstance(function, df.fem.Function)
    return function
