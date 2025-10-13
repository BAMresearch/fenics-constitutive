from __future__ import annotations

from dataclasses import dataclass

import basix
import dolfinx as df
import numpy as np
import ufl
from petsc4py import PETSc

from fenics_constitutive.solver.typesafe import typesafe


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

    def evaluate_local_incremental_gradient(
        self, cells: np.ndarray, displacement_gradient_fn: df.fem.Function
    ) -> None:
        """Eval inc disp grad fun"""
        displacement_gradient_fn.interpolate(
            self._expr,
            cells0=cells,
            cells1=np.arange(cells.size, dtype=np.int32),
        )
        displacement_gradient_fn.x.scatter_forward()


class IncrementalStress:
    __slots__ = ("_current", "_previous")

    def __init__(self, function_space) -> None:
        self._current = typesafe.fn_for(function_space)
        self._previous = typesafe.fn_for(function_space)

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
