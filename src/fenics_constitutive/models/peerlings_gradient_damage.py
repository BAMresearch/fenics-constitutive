from __future__ import annotations

import numpy as np

from .interfaces import (
    IncrSmallStrainGradientModel,
    NonlocalTangents,
    StressStrainConstraint,
)
from .utils import get_elastic_tangent, strain_from_grad_u


class PeerlingsGradientDamage(IncrSmallStrainGradientModel):
    
    def __init__(self, parameters: dict[str,float], constraint: StressStrainConstraint):
        self._constraint = constraint
        self.C = get_elastic_tangent(parameters["E"], parameters["nu"], constraint)

    
    def evaluate(
        self,
        t: float,
        del_t: float,
        grad_del_u: np.ndarray,
        nonlocal_quantity: np.ndarray,
        stress: np.ndarray,
        local_quantity: np.ndarray,
        tangents: NonlocalTangents | None,
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
        assert history is not None

        total_strain = strain_from_grad_u(grad_del_u, self._constraint) + history["total_strain"].reshape(-1, self.stress_strain_dim)

        def omega(eps_eq: np.ndarray)->np.ndarray:
            return eps_eq

        omega_new = omega(nonlocal_quantity)
        history["omega"][:] = np.maximum(history["omega"], omega_new)

        stress[:] = (1.0-history["omega"]) * total_strain @ self.C
        
        local_quantity[:] = np.linalg.norm(total_strain, axis=0)

    @property
    def constraint(self) -> StressStrainConstraint:
        """
        The constraint for the stresses or the strains.

        Returns:
            The constraint.
        """
        return self._constraint

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
    def history_dim(self) -> dict[str, int | tuple[int, int]]:
        """
        The dimensions of history variable(s). This is needed to tell the solver which quadrature
        spaces or arrays to build. If it is not none, a dictionary is returned with the name of the
        history variable as key and the dimension of the history variable as value.

        Returns:
            The dimension of the history variable(s).
        """
        return {"total_strain": self.stress_strain_dim, "omega": 1}
