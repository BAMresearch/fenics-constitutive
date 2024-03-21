from __future__ import annotations

import numpy as np

from .interfaces import Constraint

__all__ = ["strain_from_grad_u"]


def strain_from_grad_u(grad_u: np.ndarray, constraint: Constraint) -> np.ndarray:
    """Compute the Mandel-strain from the gradient of displacement.

    Parameters:
        grad_u: Gradient of displacement field.

    Returns:
        Strain tensor.
    """
    strain_dim = constraint.stress_strain_dim()
    n_gauss = int(grad_u.size / (constraint.geometric_dim() ** 2))
    strain = np.zeros(strain_dim * n_gauss)
    grad_u_view = grad_u.reshape(-1, constraint.geometric_dim() ** 2)
    strain_view = strain.reshape(-1, strain_dim)

    match constraint:
        case Constraint.UNIAXIAL_STRAIN:
            strain_view[:, 0] = grad_u_view[:, 0]
        case Constraint.UNIAXIAL_STRESS:
            strain_view[:, 0] = grad_u_view[:, 0]
        case Constraint.PLANE_STRAIN:
            """
            Full tensor:
            
            0 1
            2 3 

            Mandel vector:
            f = 1 / sqrt(2) 
            0 3 "0" f*(1+2)
            """
            strain_view[:, 0] = grad_u_view[:, 0]
            strain_view[:, 1] = grad_u_view[:, 3]
            strain_view[:, 2] = 0.0
            strain_view[:, 3] = 1 / 2**0.5 * (grad_u_view[:, 1] + grad_u_view[:, 2])
        case Constraint.PLANE_STRESS:
            """
            Full tensor:
            
            0 1
            2 3 

            Mandel vector:
            f = 1 / sqrt(2) 
            0 3 "0" f*(1+2)
            """
            strain_view[:, 0] = grad_u_view[:, 0]
            strain_view[:, 1] = grad_u_view[:, 3]
            strain_view[:, 2] = 0.0
            strain_view[:, 3] = 1 / 2**0.5 * (grad_u_view[:, 1] + grad_u_view[:, 2])
        case Constraint.FULL:
            """
            Full tensor:
            
            0 1 2
            3 4 5
            6 7 8

            Mandel vector:
            f = 1 / sqrt(2) 
            0 4 8 f*(1+3) f*(5+7) f*(2+6)
            """
            strain_view[:, 0] = grad_u_view[:, 0]
            strain_view[:, 1] = grad_u_view[:, 4]
            strain_view[:, 2] = grad_u_view[:, 8]
            strain_view[:, 3] = 1 / 2**0.5 * (grad_u_view[:, 1] + grad_u_view[:, 3])
            strain_view[:, 4] = 1 / 2**0.5 * (grad_u_view[:, 5] + grad_u_view[:, 7])
            strain_view[:, 5] = 1 / 2**0.5 * (grad_u_view[:, 2] + grad_u_view[:, 6])
        case _:
            msg = "Only full constraint implemented"
            raise NotImplementedError(msg)
    return strain
