from __future__ import annotations

from typing import Protocol

import numpy as np

from .utils import lame_parameters
from fenics_constitutive import StressStrainConstraint


class ElasticityLaw(Protocol):
    def get_D(self, E: float, nu: float) -> np.ndarray: ...
    def get_I2(self, stress_strain_dim: int) -> np.ndarray: ...


class FullConstraintLaw(ElasticityLaw):
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

    def get_I2(self, stress_strain_dim: int) -> np.ndarray:
        I2 = np.zeros(stress_strain_dim, dtype=np.float64)
        I2[0:3] = 1.0
        return I2


class PlaneStrainLaw(ElasticityLaw):
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

    def get_I2(self, stress_strain_dim: int) -> np.ndarray:
        I2 = np.zeros(stress_strain_dim, dtype=np.float64)
        I2[0:3] = 1.0
        return I2


class PlaneStressLaw(ElasticityLaw):
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

    def get_I2(self, stress_strain_dim: int) -> np.ndarray:
        I2 = np.zeros(stress_strain_dim, dtype=np.float64)
        I2[0:2] = 1.0
        return I2


class UniaxialStrainLaw(ElasticityLaw):
    def get_D(self, E: float, nu: float) -> np.ndarray:
        C = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return np.array([[C]])

    def get_I2(self, stress_strain_dim: int) -> np.ndarray:
        I2 = np.zeros(stress_strain_dim, dtype=np.float64)
        I2[0] = 1.0
        return I2


class UniaxialStressLaw(ElasticityLaw):
    def get_D(self, E: float, nu: float) -> np.ndarray:
        _ = nu
        return np.array([[E]])

    def get_I2(self, stress_strain_dim: int) -> np.ndarray:
        I2 = np.zeros(stress_strain_dim, dtype=np.float64)
        I2[0] = 1.0
        return I2


def get_elasticity_law(constraint: StressStrainConstraint) -> ElasticityLaw:
    """Factory function to return the appropriate law object for a given constraint."""
    law_map = {
        StressStrainConstraint.FULL: FullConstraintLaw(),
        StressStrainConstraint.PLANE_STRAIN: PlaneStrainLaw(),
        StressStrainConstraint.PLANE_STRESS: PlaneStressLaw(),
        StressStrainConstraint.UNIAXIAL_STRAIN: UniaxialStrainLaw(),
        StressStrainConstraint.UNIAXIAL_STRESS: UniaxialStressLaw(),
    }
    try:
        return law_map[constraint]
    except KeyError as err:
        msg = f"Constraint {constraint} not implemented."
        raise NotImplementedError(msg) from err
