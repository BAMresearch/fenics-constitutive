from __future__ import annotations

from typing import Protocol

import numpy as np

from fenics_constitutive import (
    IncrSmallStrainModel,
    StressStrainConstraint,
    strain_from_grad_u,
)

from .utils import lame_parameters


class LinearElasticityModel(IncrSmallStrainModel):
    """
    A linear elastic material model which has been implemented for all constraints.

    Args:
        parameters: Material parameters. Must contain "E" for the Youngs modulus and "nu" for the Poisson ratio.
        constraint: Constraint type.
    """

    def __init__(
        self, parameters: dict[str, float], constraint: StressStrainConstraint
    ):
        self._constraint = constraint
        E = parameters["E"]
        nu = parameters["nu"]
        law_map = {
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
        # Call get_D with E and nu for all laws
        self.D = law.get_D(E, nu)

    def evaluate(
        self,
        t: float,
        del_t: float,
        grad_del_u: np.ndarray,
        stress: np.ndarray,
        tangent: np.ndarray,
        history: np.ndarray | dict[str, np.ndarray] | None,
    ) -> None:
        # Unused: t, del_t, history
        assert (
            grad_del_u.size // (self.geometric_dim**2)
            == stress.size // self.stress_strain_dim
            == tangent.size // (self.stress_strain_dim**2)
        )
        n_gauss = grad_del_u.size // (self.geometric_dim**2)
        mandel_view = stress.reshape(-1, self.stress_strain_dim)
        strain_increment = strain_from_grad_u(grad_del_u, self.constraint)
        mandel_view += strain_increment.reshape(-1, self.stress_strain_dim) @ self.D
        tangent[:] = np.tile(self.D.flatten(), n_gauss)

    @property
    def constraint(self) -> StressStrainConstraint:
        return self._constraint

    @property
    def history_dim(self) -> None:
        return None

    # def update(self) -> None:
    #    pass


class ElasticityConstitutiveLaw(Protocol):
    def get_D(self, E: float, nu: float) -> np.ndarray: ...


class FullConstraintLaw(ElasticityConstitutiveLaw):
    def get_D(self, E: float, nu: float) -> np.ndarray:
        mu, lam = lame_parameters(E, nu)
        return np.array(
            [
                [2.0 * mu + lam, lam, lam, 0.0, 0.0, 0.0],
                [lam, 2.0 * mu + lam, lam, 0.0, 0.0, 0.0],
                [lam, lam, 2.0 * mu + lam, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.0 * mu, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 2.0 * mu, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 2.0 * mu],
            ]
        )


class PlaneStrainLaw(ElasticityConstitutiveLaw):
    def get_D(self, E: float, nu: float) -> np.ndarray:
        mu, lam = lame_parameters(E, nu)
        return np.array(
            [
                [2.0 * mu + lam, lam, lam, 0.0],
                [lam, 2.0 * mu + lam, lam, 0.0],
                [lam, lam, 2.0 * mu + lam, 0.0],
                [0.0, 0.0, 0.0, 2.0 * mu],
            ]
        )


class PlaneStressLaw(ElasticityConstitutiveLaw):
    def get_D(self, E: float, nu: float) -> np.ndarray:
        return (
            E
            / (1 - nu**2.0)
            * np.array(
                [
                    [1.0, nu, 0.0, 0.0],
                    [nu, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, (1.0 - nu)],
                ]
            )
        )


class UniaxialStrainLaw(ElasticityConstitutiveLaw):
    def get_D(self, E: float, nu: float) -> np.ndarray:
        C = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return np.array([[C]])


class UniaxialStressLaw(ElasticityConstitutiveLaw):
    def get_D(self, E: float, nu: float) -> np.ndarray:
        return np.array([[E]])
