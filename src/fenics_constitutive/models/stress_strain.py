from __future__ import annotations

import dolfinx as df
import numpy as np
import ufl

from .interfaces import IncrSmallStrainModel, StressStrainConstraint

__all__ = [
    "PlaneStrainFrom3D",
    "UniaxialStrainFrom3D",
    "strain_from_grad_u",
    "ufl_mandel_strain",
]


def ufl_mandel_strain(
    u: ufl.core.expr.Expr, constraint: StressStrainConstraint
) -> ufl.core.expr.Expr:
    """
    Compute the strain in Mandel notation from the displacement field.

    Args:
        u: Displacement field.
        constraint: Constraint that the model is implemented for.

    Returns:
        Vector-valued UFL expression of the strain in Mandel notation.
    """
    shape = len(u.ufl_shape)
    geometric_dim = u.ufl_shape[0] if shape > 0 else 1
    assert geometric_dim == constraint.geometric_dim
    match constraint:
        case StressStrainConstraint.UNIAXIAL_STRAIN:
            return ufl.nabla_grad(u)
        case StressStrainConstraint.UNIAXIAL_STRESS:
            return ufl.nabla_grad(u)
        case StressStrainConstraint.PLANE_STRAIN:
            grad_u = ufl.nabla_grad(u)
            return ufl.as_vector(
                [
                    grad_u[0, 0],
                    grad_u[1, 1],
                    0.0,
                    1 / 2**0.5 * (grad_u[0, 1] + grad_u[1, 0]),
                ]
            )
        case StressStrainConstraint.PLANE_STRESS:
            grad_u = ufl.nabla_grad(u)
            return ufl.as_vector(
                [
                    grad_u[0, 0],
                    grad_u[1, 1],
                    0.0,
                    1 / 2**0.5 * (grad_u[0, 1] + grad_u[1, 0]),
                ]
            )
        case StressStrainConstraint.FULL:
            grad_u = ufl.nabla_grad(u)
            return ufl.as_vector(
                [
                    grad_u[0, 0],
                    grad_u[1, 1],
                    grad_u[2, 2],
                    1 / 2**0.5 * (grad_u[0, 1] + grad_u[1, 0]),
                    1 / 2**0.5 * (grad_u[0, 2] + grad_u[2, 0]),
                    1 / 2**0.5 * (grad_u[1, 2] + grad_u[2, 1]),
                ]
            )

