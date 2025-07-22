from __future__ import annotations

import numpy as np

from fenics_constitutive.interfaces import StressStrainConstraint

UNIAXIAL_CONSTRAINTS = (
    StressStrainConstraint.UNIAXIAL_STRESS,
    StressStrainConstraint.UNIAXIAL_STRAIN,
)


def get_identity_submatrix(constraint: StressStrainConstraint) -> np.ndarray:
    """Returns main diagonal of the identity matrix with zero'd entries based on the stress-strain constraint."""
    result = np.zeros(constraint.stress_strain_dim, dtype=np.float64)
    if constraint in (StressStrainConstraint.FULL, StressStrainConstraint.PLANE_STRAIN):
        result[0:3] = 1.0
    elif constraint == StressStrainConstraint.PLANE_STRESS:
        result[0:2] = 1.0
    elif constraint in UNIAXIAL_CONSTRAINTS:
        result[0] = 1.0
    else:
        msg = f"Constraint {constraint} not implemented."
        raise NotImplementedError(msg)

    return result
