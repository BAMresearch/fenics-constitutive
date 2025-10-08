from __future__ import annotations

import numpy as np

from fenics_constitutive import StressStrainConstraint


def lame_parameters(E: float, nu: float) -> tuple[float, float]:
    """Compute Lame parameters (mu, lam) from Young's modulus and Poisson's ratio."""
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam

def get_elastic_tangent(E: float, nu: float, constraint: StressStrainConstraint) -> np.ndarray:
    """Get the linear elastic tangent based on the stress-strain constraint.
    Args:
        E (float): Young's modulus.
        nu (float): Poisson's ratio.
        constraint (StressStrainConstraint): Stress-strain constraint (.FULL, PLANE_STRAIN, PLANE_STRESS, UNIAXIAL_STRAIN, UNIAXIAL_STRESS).
    
    Returns:
        np.ndarray: Elastic tangent matrix.
    """

    mu, lam = lame_parameters(E, nu)
    match constraint:
        case StressStrainConstraint.FULL:
            # see https://en.wikipedia.org/wiki/Hooke%27s_law
            D = np.array(
                [
                    [2.0 * mu + lam, lam, lam, 0.0, 0.0, 0.0],
                    [lam, 2.0 * mu + lam, lam, 0.0, 0.0, 0.0],
                    [lam, lam, 2.0 * mu + lam, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 2.0 * mu, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 2.0 * mu, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 2.0 * mu],
                ]
            )
        case StressStrainConstraint.PLANE_STRAIN:
            # We assert that the strain is being provided with 0 in the z-direction
            # see https://en.wikipedia.org/wiki/Hooke%27s_law
            D = np.array(
            [
                [2.0 * mu + lam, lam, lam, 0.0],
                [lam, 2.0 * mu + lam, lam, 0.0],
                [lam, lam, 2.0 * mu + lam, 0.0],
                [0.0, 0.0, 0.0, 2.0 * mu],
            ]
            )
        case StressStrainConstraint.PLANE_STRESS:
            # We do not make any assumptions about strain in the z-direction
            # This matrix just multiplies the z component by 0.0 which results
            # in a plane stress state
            # see https://en.wikipedia.org/wiki/Hooke%27s_law
            D = E / (1 - nu**2.0) * np.array(
                [
                    [1.0, nu, 0.0, 0.0],
                    [nu, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, (1.0 - nu)],
                ]
            )
        case StressStrainConstraint.UNIAXIAL_STRAIN:
            # see https://csmbrannon.net/2012/08/02/distinction-between-uniaxial-stress-and-uniaxial-strain/
            C = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
            D = np.array([[C]])

        case StressStrainConstraint.UNIAXIAL_STRESS:
            # see https://csmbrannon.net/2012/08/02/distinction-between-uniaxial-stress-and-uniaxial-strain/
            D = np.array([[E]])

        case _:
            msg = "Constraint not implemented"
            raise NotImplementedError(msg)

    return D


def get_identity(stress_strain_dim: int, constraint: StressStrainConstraint) -> np.ndarray:
    """Get the identity tensor based on the stress-strain constraint.
    Args:
        stress_strain_dim (int): Dimension of the stress/strain vector.
        constraint (StressStrainConstraint): Stress-strain constraint (.FULL, PLANE_STRAIN, PLANE_STRESS, UNIAXIAL_STRAIN, UNIAXIAL_STRESS)
    Returns:
        np.ndarray: Identity tensor.
    """
 
    match constraint:
        case StressStrainConstraint.FULL:
            I2 = np.zeros(stress_strain_dim, dtype=np.float64)
            I2[0:3] = 1.0
        case StressStrainConstraint.PLANE_STRAIN:
            # We assert that the strain is being provided with 0 in the z-direction
            I2 = np.zeros(stress_strain_dim, dtype=np.float64)
            I2[0:3] = 1.0
        case StressStrainConstraint.PLANE_STRESS:
            I2 = np.zeros(stress_strain_dim, dtype=np.float64)
            I2[0:2] = 1.0
        case StressStrainConstraint.UNIAXIAL_STRAIN:
            I2 = np.zeros(stress_strain_dim, dtype=np.float64)
            I2[0] = 1.0
        case StressStrainConstraint.UNIAXIAL_STRESS:
            I2 = np.zeros(stress_strain_dim, dtype=np.float64)
            I2[0] = 1.0

        case _:
            msg = "Constraint not implemented"
            raise NotImplementedError(msg)

    return I2