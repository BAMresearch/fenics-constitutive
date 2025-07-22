from __future__ import annotations

from typing import Protocol

import numpy as np

from fenics_constitutive import StressStrainConstraint

from .utils import lame_parameters


class IsotropicMaterialTensor(Protocol):
    def get(self, E: float, nu: float) -> np.ndarray: ...


class FullIsotropicTensor(IsotropicMaterialTensor):
    def get(self, E: float, nu: float) -> np.ndarray:
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


class PlaneStrainIsotropicTensor(IsotropicMaterialTensor):
    def get(self, E: float, nu: float) -> np.ndarray:
        mu, lam = lame_parameters(E, nu)
        return np.array(
            [
                [2.0 * mu + lam, lam, lam, 0.0],
                [lam, 2.0 * mu + lam, lam, 0.0],
                [lam, lam, 2.0 * mu + lam, 0.0],
                [0.0, 0.0, 0.0, 2.0 * mu],
            ]
        )


class PlaneStressIsotropicTensor(IsotropicMaterialTensor):
    def get(self, E: float, nu: float) -> np.ndarray:
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


class UniaxialStrainIsotropicTensor(IsotropicMaterialTensor):
    def get(self, E: float, nu: float) -> np.ndarray:
        C = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return np.array([[C]])


class UniaxialStressIsotropicTensor(IsotropicMaterialTensor):
    def get(self, E: float, nu: float) -> np.ndarray:
        _ = nu
        return np.array([[E]])


def get_material_tensor(constraint: StressStrainConstraint) -> IsotropicMaterialTensor:
    """Factory function to return the appropriate law object for a given constraint."""
    tensor_map = {
        StressStrainConstraint.FULL: FullIsotropicTensor(),
        StressStrainConstraint.PLANE_STRAIN: PlaneStrainIsotropicTensor(),
        StressStrainConstraint.PLANE_STRESS: PlaneStressIsotropicTensor(),
        StressStrainConstraint.UNIAXIAL_STRAIN: UniaxialStrainIsotropicTensor(),
        StressStrainConstraint.UNIAXIAL_STRESS: UniaxialStressIsotropicTensor(),
    }
    try:
        return tensor_map[constraint]
    except KeyError as err:
        msg = f"Constraint {constraint} not implemented."
        raise NotImplementedError(msg) from err
