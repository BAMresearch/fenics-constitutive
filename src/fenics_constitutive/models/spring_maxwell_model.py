from __future__ import annotations

import numpy as np

from fenics_constitutive import (
    IncrSmallStrainModel,
    StressStrainConstraint,
    strain_from_grad_u,
)

from .elasticity_laws import (
    ElasticityLaw,
    FullConstraintLaw,
    PlaneStrainLaw,
    PlaneStressLaw,
    UniaxialStrainLaw,
    UniaxialStressLaw,
)
from .utils import lame_parameters


class SpringMaxwellModel(IncrSmallStrainModel):
    """viscoelastic model based on 1D Three Parameter Model with spring and Maxwell body in parallel

             |----------- E_0: spring  ----------|
           --|                                   |--
             |--- E_1: spring --- eta: damper ---|

    with deviatoric assumptions for 3D generalization (volumetric part of visco strain == 0 damper just working on deviatoric part)
    time integration: backward Euler

    Args:
        parameters: Material parameters. Must contain "E0" for the elastic Youngs modulus, "E1" for the viscous modulus and "tau" for the relaxation time.
        constraint: Constraint type.

    """

    def __init__(
        self, parameters: dict[str, float], constraint: StressStrainConstraint
    ):
        self._constraint = constraint
        self.E0 = parameters["E0"]  # elastic modulus
        self.E1 = parameters["E1"]  # visco modulus
        self.tau = parameters["tau"]  # relaxation time
        if constraint == StressStrainConstraint.UNIAXIAL_STRESS:
            self.nu = 0.0
        else:
            self.nu = parameters["nu"]  # Poisson's ratio

        law_map: dict[StressStrainConstraint, ElasticityLaw] = {
            StressStrainConstraint.FULL: FullConstraintLaw(),
            StressStrainConstraint.PLANE_STRAIN: PlaneStrainLaw(),
            StressStrainConstraint.PLANE_STRESS: PlaneStressLaw(),
            StressStrainConstraint.UNIAXIAL_STRAIN: UniaxialStrainLaw(),
            StressStrainConstraint.UNIAXIAL_STRESS: UniaxialStressLaw(),
        }
        try:
            law = law_map[constraint]
        except KeyError as err:
            msg = "Constraint not implemented"
            raise NotImplementedError(msg) from err
        self.D_0 = law.get_D(self.E0, self.nu)
        self.D_1 = law.get_D(self.E1, self.nu)

    def evaluate(
        self,
        t: float,
        del_t: float,
        grad_del_u: np.ndarray,
        stress: np.ndarray,
        tangent: np.ndarray,
        history: np.ndarray | dict[str, np.ndarray] | None,
    ) -> None:
        _ = t
        assert (
            grad_del_u.size // (self.geometric_dim**2)
            == stress.size // self.stress_strain_dim
            == tangent.size // (self.stress_strain_dim**2)
        )

        # reshape gauss point arrays
        n_gauss = grad_del_u.size // (self.geometric_dim**2)
        mandel_view = stress.reshape(-1, self.stress_strain_dim)

        strain_increment = strain_from_grad_u(grad_del_u, self.constraint).reshape(
            -1, self.stress_strain_dim
        )
        if history is None:
            msg = "history must not be None"
            raise ValueError(msg)
        strain_visco_n = history["strain_visco"].reshape(-1, self.stress_strain_dim)
        strain_n = history["strain"].reshape(-1, self.stress_strain_dim)

        assert del_t > 0, "Time step must be defined and positive."

        mu1, _ = lame_parameters(self.E1, self.nu)

        strain_total = strain_n + strain_increment
        factor = 1 / del_t + 1 / self.tau
        _deps_visko = (
            1
            / factor
            * (
                1 / (self.tau * 2 * mu1) * strain_total @ self.D_1
                - 1 / self.tau * strain_visco_n
            )
        )

        dstress = strain_increment @ (self.D_0 + self.D_1) - 2 * mu1 * _deps_visko
        mandel_view += dstress
        D = self.D_0 + (1 - 1 / (self.tau * factor)) * self.D_1

        tangent[:] = np.tile(D.flatten(), n_gauss)
        strain_visco_n += _deps_visko
        strain_n += strain_increment

    @property
    def constraint(self) -> StressStrainConstraint:
        return self._constraint

    @property
    def history_dim(self) -> dict[str, int | tuple[int, int]] | None:
        return {
            "strain_visco": self.stress_strain_dim,
            "strain": self.stress_strain_dim,
        }

    # def update(self) -> None:
    #    pass
