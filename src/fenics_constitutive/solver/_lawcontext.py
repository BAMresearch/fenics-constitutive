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
    from ._solver import DisplacementGradientFunction, IncrSmallStrainProblem


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
