from fenics_constitutive.models.interfaces import NonlocalTangents
from typing import cast
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import dolfinx as df
import numpy as np

from fenics_constitutive.models.interfaces import (
    IncrSmallStrainGradientModel,
    IncrSmallStrainModel,
)
from fenics_constitutive.solver._incrementalunknowns import IncrementalGradientSolution
from fenics_constitutive.solver._spaces import GradientElements

from ._history import History
from ._incrementalunknowns import IncrementalDisplacement, IncrementalStress, IncrementalLocalQuantity
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
    tangent_fn: df.fem.Function = fn_for(
        element_spaces.stress_tensor_space(submesh)
    )
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
    law: IncrSmallStrainGradientModel, cells: np.ndarray, elements: GradientElements, stress_space: df.fem.FunctionSpace
) -> GradientLawOnSubMesh:
    """Create a `LawOnSubMesh`"""
    subspace_map, submesh, stress_vector_space = build_subspace_map(
        cells, stress_space,
    )
    stress_fn = fn_for(stress_vector_space)
    local_nonlocal_space = elements.local_quantity_space(submesh)
    local_fn = fn_for(local_nonlocal_space)
    nonlocal_fn = fn_for(local_nonlocal_space)
    tangent_spaces = elements.tangent_spaces(submesh)
    tangents = [cast(df.fem.Function,df.fem.Function(space)) for space in tangent_spaces]
    
    inc_disp_grad_fn = fn_for(
        elements.del_grad_u_space(submesh)
    )

    history = History.try_create(law, submesh, elements.q_degree)
    return GradientLawOnSubMesh(
        law=law,
        cells=cells,
        displacement_gradient_fn=inc_disp_grad_fn,
        nonlocal_quantity_sub=nonlocal_fn,
        local_quantity_sub=local_fn,
        stress=stress_fn,
        dsigma_deps_sub=tangents[0],
        dsigma_dnonlocal_sub=tangents[1],
        dlocal_deps_sub=tangents[2],
        dlocal_dnonlocal_sub=tangents[3],
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
    dsigma_deps_sub: df.fem.Function
    dsigma_dnonlocal_sub: df.fem.Function
    dlocal_deps_sub: df.fem.Function
    dlocal_dnonlocal_sub: df.fem.Function
    submesh_map: SpaceMap
    history: History | None = None
    
    
    def stress_sub(self, stress: IncrementalStress) -> np.ndarray:
        """Map the global stress to the submesh"""
        self.submesh_map.map_to_sub(stress.previous, self.stress)
        return self.stress.x.array

    def map_to_parent(
        self,
        global_stress: IncrementalStress,
        global_dsigma_deps: df.fem.Function,
        global_dsigma_dnonlocal: df.fem.Function,
        global_dlocal_deps: df.fem.Function,
        global_dlocal_dnonlocal: df.fem.Function,
    ) -> None:
        """Map stresses and tangents back to the main mesh"""
        self.submesh_map.map_to_parent(self.stress, global_stress.current)
        self.submesh_map.map_to_parent(self.dsigma_deps_sub, global_dsigma_deps)
        self.submesh_map.map_to_parent(self.dsigma_dnonlocal_sub, global_dsigma_dnonlocal)
        self.submesh_map.map_to_parent(self.dlocal_deps_sub, global_dlocal_deps)
        self.submesh_map.map_to_parent(self.dlocal_dnonlocal_sub, global_dlocal_dnonlocal)

    def evaluate(
        self,
        sim_time: SimulationTime,
        incr_solution: IncrementalGradientSolution,
        global_stress: IncrementalStress,
        global_tangent: df.fem.Function,
    ) -> None:
        """Perform a full constitutive model evaluation for this law context."""
        incr_solution.evaluate_local_incremental_gradient(
            self.cells, self.displacement_gradient_fn
        )
        incr_solution.evaluate_nonlocal_on_quadrature_points(self.cells, self.nonlocal_quantity_sub)
        history_input = (
            self.history.reset_trial_state() if self.history is not None else None
        )
        tangent = NonlocalTangents(dsigma_deps=self.dsigma_deps_sub.x.array, dsigma_dnonlocal=self.dsigma_dnonlocal_sub.x.array, dlocal_deps=self.dlocal_deps_sub.x.array, dlocal_dnonlocal=self.dlocal_dnonlocal_sub.x.array)
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
        self.map_to_parent(global_stress,)

    def update_history(self) -> None:
        """Update the history for this law context if it exists."""
        if self.history is not None:
            self.history.update()