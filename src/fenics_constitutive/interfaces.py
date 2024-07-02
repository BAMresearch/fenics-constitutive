from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from enum import Enum

import numpy as np

__all__ = [
    "Constraint",
    "IncrSmallStrainModel",
]


class Constraint(Enum):
    """
    Enum for the model constraint.

    The constraint can either be:
    - UNIAXIAL_STRAIN
    - UNIAXIAL_STRESS
    - PLANE_STRAIN
    - PLANE_STRESS
    - FULL

    """

    UNIAXIAL_STRAIN = 1
    UNIAXIAL_STRESS = 2
    PLANE_STRAIN = 3
    PLANE_STRESS = 4
    FULL = 5

    def stress_strain_dim(self) -> int:
        """
        The stress-strain dimension of the constraint.

        Returns:
            The stress-strain dimension.
        """
        match self:
            case Constraint.UNIAXIAL_STRAIN:
                return 1
            case Constraint.UNIAXIAL_STRESS:
                return 1
            case Constraint.PLANE_STRAIN:
                return 4
            case Constraint.PLANE_STRESS:
                return 4
            case Constraint.FULL:
                return 6

    def geometric_dim(self) -> int:
        """
        The geometric dimension for the constraint.

        Returns:
            The geometric dimension.
        """
        match self:
            case Constraint.UNIAXIAL_STRAIN:
                return 1
            case Constraint.UNIAXIAL_STRESS:
                return 1
            case Constraint.PLANE_STRAIN:
                return 2
            case Constraint.PLANE_STRESS:
                return 2
            case Constraint.FULL:
                return 3


class IncrSmallStrainModel(ABC):
    """
    Interface for incremental small strain models.
    """

    @abstractmethod
    def evaluate(
        self,
        time: float,
        del_t: float,
        grad_del_u: np.ndarray,
        mandel_stress: np.ndarray,
        tangent: np.ndarray,
        history: np.ndarray | dict[str, np.ndarray] | None,
    ) -> None:
        """
        Evaluate the constitutive model and overwrite the stress, tangent and history.

        Args:
            time : The current global time.
            del_t : The time increment.
            grad_del_u : The gradient of the increment of the displacement field.
            mandel_stress : The Mandel stress.
            tangent : The tangent.
            history : The history variable(s).
        """

    @abstractmethod
    def update(self) -> None:
        """
        Called after the solver has converged to update anything that is not
        contained in the history variable(s).
        For example: The model could contain the current time which is not
        stored in the history, but needs to be updated after each evaluation.
        """

    @abstractproperty
    def constraint(self) -> Constraint:
        """
        The constraint that the model is implemented for.

        Returns:
            The constraint.
        """

    @property
    def stress_strain_dim(self) -> int:
        """
        The stress-strain dimension that the model is implemented for.

        Returns:
            The stress-strain dimension.
        """
        return self.constraint.stress_strain_dim()

    @property
    def geometric_dim(self) -> int:
        """
        The geometric dimension that the model is implemented for.

        Returns:
            The geometric dimension.
        """
        return self.constraint.geometric_dim()

    @abstractproperty
    def history_dim(self) -> dict[str, int | tuple[int, int]] | None:
        """
        The dimensions of history variable(s). This is needed to tell the solver which quadrature
        spaces or arrays to build. If all history variables are stored in a single
        array, then the dimension of the array is returned. If the history variables are stored
        in seperate arrays (or functions), then a dictionary is returned with the name of the
        history variable as key and the dimension of the history variable as value.

        Returns:
            The dimension of the history variable(s).
        """
