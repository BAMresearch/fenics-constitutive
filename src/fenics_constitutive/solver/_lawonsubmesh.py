from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import dolfinx as df
import numpy as np

from fenics_constitutive.models.interfaces import (
    IncrSmallStrainGradientModel,
    IncrSmallStrainModel,
    NonlocalTangentFunctions,
    NonlocalTangents,
)
from fenics_constitutive.solver._incrementalunknowns import IncrementalGradientSolution
from fenics_constitutive.solver._spaces import GradientElements

from ._history import History
from ._incrementalunknowns import (
    IncrementalDisplacement,
    IncrementalLocalQuantity,
    IncrementalStress,
)
from ._spaces import ElementSpaces
from .maps import SpaceMap, build_subspace_map
from .typesafe import fn_for

if TYPE_CHECKING:
    from ._solver import SimulationTime


def create_law_on_submesh(
    law: IncrSmallStrainModel, cells: np.ndarray, element_spaces: ElementSpaces
) -> LawOnSubMesh:
    """Create a `LawOnSubMesh`"""
    subspace_map, submesh, stress_vector_space = build_subspace_map(
        cells, element_spaces.stress_vector_space
    )
    stress_fn = fn_for(stress_vector_space)
    tangent_fn: df.fem.Function = fn_for(element_spaces.stress_tensor_space(submesh))
    inc_disp_grad_fn = fn_for(
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


@dataclass
class LawOnSubMesh:
    law: IncrSmallStrainModel
    cells: np.ndarray
    displacement_gradient_fn: df.fem.Function
    stress: df.fem.Function
    local_tangent: df.fem.Function
    submesh_map: SpaceMap
    history: History | None = None

    def local_stress(self, stress: IncrementalStress) -> np.ndarray:
        """Map the global stress to the submesh"""
        self.submesh_map.map_to_sub(stress.previous, self.stress)
        return self.stress.x.array

    def map_to_parent(
        self,
        global_stress: IncrementalStress,
        global_tangent: df.fem.Function,
    ) -> None:
        """Map stresses and tangents back to the main mesh"""
        self.submesh_map.map_to_parent(self.stress, global_stress.current)
        self.submesh_map.map_to_parent(self.local_tangent, global_tangent)

    def evaluate(
        self,
        sim_time: SimulationTime,
        incr_disp: IncrementalDisplacement,
        global_stress: IncrementalStress,
        global_tangent: df.fem.Function,
    ) -> None:
        """Perform a full constitutive model evaluation for this law context."""
        incr_disp.evaluate_local_incremental_gradient(
            self.cells, self.displacement_gradient_fn
        )
        history_input = (
            self.history.reset_trial_state() if self.history is not None else None
        )
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

    def update_history(self) -> None:
        """Update the history for this law context if it exists."""
        if self.history is not None:
            self.history.update()


def create_gradient_law_on_submesh(
    law: IncrSmallStrainGradientModel,
    cells: np.ndarray,
    elements: GradientElements,
    stress_space: df.fem.FunctionSpace,
) -> GradientLawOnSubMesh:
    """Create a `LawOnSubMesh`"""
    subspace_map, submesh, stress_vector_space = build_subspace_map(
        cells,
        stress_space,
    )
    stress_fn = fn_for(stress_vector_space)
    local_nonlocal_space = elements.local_quantity_space(submesh)
    local_fn = fn_for(local_nonlocal_space)
    nonlocal_fn = fn_for(local_nonlocal_space)
    tangent_spaces = elements.tangent_spaces(submesh)
    tangents = [
        cast(df.fem.Function, df.fem.Function(space)) for space in tangent_spaces
    ]
    tangent_functions = NonlocalTangentFunctions(
        dsigma_deps=tangents[0],
        dsigma_dnonlocal=tangents[1],
        dlocal_deps=tangents[2],
        dlocal_dnonlocal=tangents[3],
    )

    inc_disp_grad_fn = fn_for(elements.del_grad_u_space(submesh))

    history = History.try_create(law, submesh, elements.q_degree)
    return GradientLawOnSubMesh(
        law=law,
        cells=cells,
        displacement_gradient_fn=inc_disp_grad_fn,
        nonlocal_quantity_sub=nonlocal_fn,
        local_quantity_sub=local_fn,
        stress=stress_fn,
        tangents_sub=tangent_functions,
        submesh_map=subspace_map,
        history=history,
    )


@dataclass
class GradientLawOnSubMesh:
    law: IncrSmallStrainGradientModel
    cells: np.ndarray
    displacement_gradient_fn: df.fem.Function
    nonlocal_quantity_sub: df.fem.Function
    local_quantity_sub: df.fem.Function
    stress: df.fem.Function
    tangents_sub: NonlocalTangentFunctions
    submesh_map: SpaceMap
    history: History | None = None

    def stress_sub(self, stress: IncrementalStress) -> np.ndarray:
        """Map the global stress to the submesh"""
        self.submesh_map.map_to_sub(stress.previous, self.stress)
        return self.stress.x.array

    def map_to_parent(
        self,
        global_stress: IncrementalStress,
        global_tangents: NonlocalTangentFunctions,
        global_local_quantity: IncrementalLocalQuantity,
    ) -> None:
        """Map stresses and tangents back to the main mesh"""
        self.submesh_map.map_to_parent(self.stress, global_stress.current)
        self.submesh_map.map_to_parent(self.local_quantity_sub, global_local_quantity.current)

        self.submesh_map.map_to_parent(self.tangents_sub.dsigma_deps, global_tangents.dsigma_deps)
        self.submesh_map.map_to_parent(self.tangents_sub.dsigma_dnonlocal, global_tangents.dsigma_dnonlocal)
        self.submesh_map.map_to_parent(self.tangents_sub.dlocal_deps, global_tangents.dlocal_deps)
        self.submesh_map.map_to_parent(self.tangents_sub.dlocal_dnonlocal, global_tangents.dlocal_dnonlocal)

    def evaluate(
        self,
        sim_time: SimulationTime,
        incr_solution: IncrementalGradientSolution,
        global_stress: IncrementalStress,
        global_local_quantity: IncrementalLocalQuantity,
        global_tangents: NonlocalTangentFunctions,

    ) -> None:
        """Perform a full constitutive model evaluation for this law context."""
        incr_solution.evaluate_local_incremental_gradient(
            self.cells, self.displacement_gradient_fn
        )
        incr_solution.evaluate_nonlocal_on_quadrature_points(
            self.cells, self.nonlocal_quantity_sub
        )
        history_input = (
            self.history.reset_trial_state() if self.history is not None else None
        )
        tangent = self.tangents_sub.to_nonlocal_tangents()
        with df.common.Timer("constitutive-law-evaluation"):
            self.law.evaluate(
                sim_time.current,
                sim_time.dt,
                self.displacement_gradient_fn.x.array,
                self.nonlocal_quantity_sub.x.array,
                self.stress_sub(global_stress),
                self.local_quantity_sub.x.array,
                tangent,
                history_input,
            )
        self.map_to_parent(
            global_stress, global_tangents, global_local_quantity,
        )

    def update_history(self) -> None:
        """Update the history for this law context if it exists."""
        if self.history is not None:
            self.history.update()
