from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import dolfinx as df
import numpy as np

from fenics_constitutive import typesafe
from fenics_constitutive.interfaces import IncrSmallStrainModel
from fenics_constitutive.maps import SpaceMap, build_subspace_map

from ._history import History
from ._incrementalunknowns import IncrementalDisplacement, IncrementalStress
from ._spaces import ElementSpaces

if TYPE_CHECKING:
    from ._solver import SimulationTime


@dataclass
class LawOnSubMesh:
    law: IncrSmallStrainModel
    cells: np.ndarray
    displacement_gradient_fn: df.fem.Function
    stress: df.fem.Function
    local_tangent: df.fem.Function
    submesh_map: SpaceMap
    history: History | None = None

    @staticmethod
    def map_to_cells(
        law: IncrSmallStrainModel, cells: np.ndarray, element_spaces: ElementSpaces
    ) -> LawOnSubMesh:
        subspace_map, submesh, stress_vector_space = build_subspace_map(
            cells, element_spaces.stress_vector_space
        )
        stress_fn = typesafe.fn_for(stress_vector_space)
        tangent_fn: df.fem.Function = typesafe.fn_for(
            element_spaces.stress_tensor_space(submesh)
        )
        inc_disp_grad_fn = typesafe.fn_for(
            element_spaces.displacement_gradient_tensor_space(submesh)
        )

        history = History.try_create(law, submesh, element_spaces.q_degree)
        return LawOnSubMesh(
            law=law,
            cells=cells,
            displacement_gradient_fn=inc_disp_grad_fn,
            stress=stress_fn,
            local_tangent=tangent_fn,
            submesh_map=subspace_map,
            history=history,
        )

    def local_stress(self, stress: IncrementalStress) -> np.ndarray:
        self.submesh_map.map_to_child(stress.previous, self.stress)
        return self.stress.x.array

    def map_to_parent(
        self,
        global_stress: IncrementalStress,
        global_tangent: df.fem.Function,
    ) -> None:
        self.submesh_map.map_to_parent(self.stress, global_stress.current)
        self.submesh_map.map_to_parent(self.local_tangent, global_tangent)

    def evaluate(
        self,
        sim_time: SimulationTime,
        incr_disp: IncrementalDisplacement,
        global_stress: IncrementalStress,
        global_tangent: df.fem.Function,
    ) -> None:
        """Perform a full constitutive update for this law context."""
        incr_disp.evaluate_local_incremental_gradient(
            self.cells, self.displacement_gradient_fn
        )
        history_input = self.history.advance() if self.history is not None else None
        with df.common.Timer("constitutive-law-evaluation"):
            self.law.evaluate(
                sim_time.current,
                sim_time.dt,
                self.displacement_gradient_fn.x.array,
                self.local_stress(global_stress),
                self.local_tangent.x.array,
                history_input,
            )
        self.map_to_parent(global_stress, global_tangent)

    def commit_history(self) -> None:
        """Commit the history for this law context if it exists."""
        if self.history is not None:
            self.history.commit()
