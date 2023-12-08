import numpy as np
from abc import ABC, abstractmethod, abstractproperty
from enum import Enum


__all__ = [
    "Constraint",
    "IncrSmallStrainModel",
]


class Constraint(Enum):
    UNIAXIAL_STRAIN = 1
    UNIAXIAL_STRESS = 2
    PLANE_STRAIN = 3
    PLANE_STRESS = 4
    FULL = 5

    def stress_strain_dim(self):
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


class IncrSmallStrainModel(ABC):
    """Interface for incremental small strain models."""

    @abstractmethod
    def evaluate(
        self, del_t: float, grad_u: np.ndarray, mandel_stress: np.ndarray, tangent: np.ndarray
    ) -> None:
        """Evaluate the constitutive model.

        Parameters:
            grad_u : The gradient of the displacement field.
            mandel_stress : The Mandel stress.
            tangent : The tangent.
        """
        pass

    @abstractproperty
    def constraint(self) -> Constraint:
        """The constraint."""
        pass
