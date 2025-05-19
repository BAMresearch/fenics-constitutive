from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import dolfinx as df
import numpy as np

from fenics_constitutive.interfaces import IncrSmallStrainModel
from fenics_constitutive.maps import SubSpaceMap

from ._history import History

if TYPE_CHECKING:
    from ._solver import (
        DisplacementGradientFunction,
        IncrementalDisplacement,
        IncrementalStress,
        SimulationTime,
    )


class LawContext(ABC):
    law: IncrSmallStrainModel
    displacement_gradient: DisplacementGradientFunction
    history: History | None

    @abstractmethod
    def update_stress_and_tangent(
        self,
        stress: IncrementalStress,
        tangent: df.fem.Function,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @abstractmethod
    def map_to_parent(
        self,
        stress: IncrementalStress,
        tangent: df.fem.Function,
    ) -> None: ...

    def displacement_gradient_array(self) -> np.ndarray:
        return self.displacement_gradient.displacement_gradient()

    def scatter_displacement_gradient(self) -> None:
        self.displacement_gradient.scatter()

    def step(
        self,
        sim_time: SimulationTime,
        incr_disp: IncrementalDisplacement,
        stress: IncrementalStress,
        tangent: df.fem.Function,
    ) -> None:
        """Perform a full constitutive update for this law context."""
        incr_disp.evaluate_incremental_gradient(self.displacement_gradient)
        stress_input, tangent_input = self.update_stress_and_tangent(stress, tangent)
        history_input = self.history.advance() if self.history is not None else None
        with df.common.Timer("constitutive-law-evaluation"):
            self.law.evaluate(
                sim_time.current,
                sim_time.dt,
                self.displacement_gradient.displacement_gradient_fn.x.array,
                stress_input,
                tangent_input,
                history_input,
            )
        self.map_to_parent(stress, tangent)

    def commit_history(self) -> None:
        """Commit the history for this law context if it exists."""
        if self.history is not None:
            self.history.commit()


@dataclass
class SingleLawContext(LawContext):
    law: IncrSmallStrainModel
    displacement_gradient: DisplacementGradientFunction
    history: History | None = None

    def update_stress_and_tangent(
        self,
        stress: IncrementalStress,
        tangent: df.fem.Function,
    ) -> tuple[np.ndarray, np.ndarray]:
        stress.update_current()
        return stress.current_array(), tangent.x.array

    def map_to_parent(
        self,
        stress: IncrementalStress,
        tangent: df.fem.Function,
    ) -> None:
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
        self,
        stress: IncrementalStress,
        tangent: df.fem.Function,  # noqa: ARG002
    ) -> tuple[np.ndarray, np.ndarray]:
        self.submesh_map.map_to_child(stress.previous, self.stress)
        return self.stress.x.array, self.tangent.x.array

    def map_to_parent(
        self,
        stress: IncrementalStress,
        tangent: df.fem.Function,
    ) -> None:
        self.submesh_map.map_to_parent(self.stress, stress.current)
        self.submesh_map.map_to_parent(self.tangent, tangent)
