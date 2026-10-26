from __future__ import annotations

import numpy as np

from .interfaces import (
    IncrSmallStrainGradientModel,
    NonlocalTangents,
    StressStrainConstraint,
)
from .utils import get_elastic_tangent, strain_from_grad_u


class PeerlingsGradientPerfectDamage(IncrSmallStrainGradientModel):
    """
    Peerlings gradient damage model with perfect damage behavior.
    The model is based on the paper "GRADIENT ENHANCED DAMAGE FOR QUASI-BRITTLE MATERIALS" by Peerlings et al. (1996).

    Attributes:
        C: The elastic tangent stiffness matrix.
        eps_0: The strain at which damage initiates.
        omega_max: The maximum damage value. This should be less than 1.0 to avoid complete loss of stiffness, which can lead to numerical issues.

    Args:
        parameters: A dictionary containing the parameters of the model. It should have
            the following keys:
                - E: The Young's modulus.
                - nu: The Poisson's ratio.
                - eps_0: The strain at which damage initiates.
                - omega_max: The maximum damage value.
        constraint: The constraint for the stresses and the strains. The model should work for all assumptions,
            except maybe for plane stress. So far it is only tested for uniaxial stress.
    """

    def __init__(
        self, parameters: dict[str, float], constraint: StressStrainConstraint
    ):
        self._constraint = constraint
        self.C = get_elastic_tangent(parameters["E"], parameters["nu"], constraint)
        self.eps_0 = parameters["eps_0"]
        self.omega_max = parameters["omega_max"]

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
        ngauss = grad_del_u.size // self.geometric_dim**2
        total_strain = (
            strain_from_grad_u(grad_del_u, self._constraint) + history["total_strain"]
        ).reshape(-1, self.stress_strain_dim)
        history["total_strain"][:] = total_strain.flatten()

        zeros = np.zeros(ngauss)

        def omega(eps_eq: np.ndarray) -> np.ndarray:
            # perfect damage law
            mask = eps_eq >= self.eps_0
            damage = zeros.copy()
            damage[mask] = (1 - self.eps_0 / eps_eq[mask]) * self.omega_max
            return damage

        def strain_norm(total_strain: np.ndarray) -> np.ndarray:
            # euclidian norm
            return np.linalg.norm(total_strain, axis=1)

        omega_new = omega(nonlocal_quantity)
        omega_old = history["omega"].copy()
        history["omega"][:] = np.maximum(omega_old, omega_new)
        history_view = history["omega"].reshape(-1, 1)

        stress[:] = ((1.0 - history_view) * total_strain @ self.C).flatten()

        # this norm will work for the analytical solution from the peerlings paper
        local_quantity[:] = strain_norm(total_strain)

        if tangents is not None:
            mask = (local_quantity > 0.0)
            dlocal_deps = np.zeros((ngauss, self.stress_strain_dim))
            dlocal_deps[mask] = total_strain[mask] / local_quantity.reshape(-1, 1)[mask]
            tangents.dlocal_deps[:] = dlocal_deps.flatten()

            # This tangent is only nonzero for plasticity models
            tangents.dlocal_dnonlocal[:] = 0.0

            tangents.dsigma_deps[:] = np.tile(self.C.flatten(), ngauss)
            tangents.dsigma_deps.reshape(-1, self.stress_strain_dim**2)[:] *= (
                1.0 - history_view
            )

            # zero while damage is frozen (omega_new below the stored maximum)
            mask = (nonlocal_quantity >= self.eps_0) & (omega_new >= omega_old)
            domega_dnonlocal = zeros.copy()
            domega_dnonlocal[mask] = (self.omega_max * self.eps_0) / nonlocal_quantity[
                mask
            ] ** 2

            tangents.dsigma_dnonlocal.reshape(-1, self.stress_strain_dim)[:] = (
                -total_strain @ self.C
            ) * domega_dnonlocal.reshape(-1, 1)

    @property
    def constraint(self) -> StressStrainConstraint:
        """
        The constraint for the stresses or the strains.

        Returns
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
