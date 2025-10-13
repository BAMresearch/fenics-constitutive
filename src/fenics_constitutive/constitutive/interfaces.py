from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

__all__ = [
    "IncrSmallStrainModel",
    "StressStrainConstraint",
]


class StressStrainConstraint(Enum):
    """
    Enum for the model constraint.

    The constraint can either be: UNIAXIAL_STRAIN,
    UNIAXIAL_STRESS, PLANE_STRAIN, PLANE_STRESS, FULL

    """

    UNIAXIAL_STRAIN = 1
    UNIAXIAL_STRESS = 2
    PLANE_STRAIN = 3
    PLANE_STRESS = 4
    FULL = 5

    @property
    def stress_strain_dim(self) -> int:
        """
        The stress-strain dimension of the constraint.

        Returns:
            The stress-strain dimension.
        """
        match self:
            case StressStrainConstraint.UNIAXIAL_STRAIN:
                return 1
            case StressStrainConstraint.UNIAXIAL_STRESS:
                return 1
            case StressStrainConstraint.PLANE_STRAIN:
                return 4
            case StressStrainConstraint.PLANE_STRESS:
                return 4
            case StressStrainConstraint.FULL:
                return 6
            case _:
                msg = f"Constraint {self} not supported"
                raise Exception(msg)

    @property
    def geometric_dim(self) -> int:
        """
        The geometric dimension for the constraint.

        Returns:
            The geometric dimension.
        """
        match self:
            case StressStrainConstraint.UNIAXIAL_STRAIN:
                return 1
            case StressStrainConstraint.UNIAXIAL_STRESS:
                return 1
            case StressStrainConstraint.PLANE_STRAIN:
                return 2
            case StressStrainConstraint.PLANE_STRESS:
                return 2
            case StressStrainConstraint.FULL:
                return 3
            case _:
                msg = f"Constraint {self} not supported"
                raise Exception(msg)


class IncrSmallStrainModel(ABC):
    """
    Interface for incremental small strain models.
    """

    @abstractmethod
    def evaluate(
        self,
        t: float,
        del_t: float,
        grad_del_u: np.ndarray,
        stress: np.ndarray,
        tangent: np.ndarray,
        history: dict[str, np.ndarray] | None,
    ) -> None:
        r"""
        Evaluate the constitutive model and overwrite the stress, tangent and history.

        Args:
            t: The current global time $t_n$.
            del_t: The time increment $\Delta t$. The time at the end of the increment is $t_{n+1}=t_n+\Delta t$.
            grad_del_u: The gradient of the increment of the displacement field $\nabla\delta$ with $\delta=u_{n+1}-u_n$.
            stress: The current stress in Mandel notation.
            tangent: The tangent compatible with Mandel notation.
            history: The history variable(s).
        """

    @property
    @abstractmethod
    def constraint(self) -> StressStrainConstraint:
        """
        The constraint for the stresses or the strains.

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
        return self.constraint.stress_strain_dim

    @property
    def geometric_dim(self) -> int:
        """
        The geometric dimension that the model is implemented for.

        Returns:
            The geometric dimension.
        """
        return self.constraint.geometric_dim

    @property
    @abstractmethod
    def history_dim(self) -> dict[str, int | tuple[int, int]] | None:
        """
        The dimensions of history variable(s). This is needed to tell the solver which quadrature
        spaces or arrays to build. If it is not none, a dictionary is returned with the name of the
        history variable as key and the dimension of the history variable as value.

        Returns:
            The dimension of the history variable(s).
        """
