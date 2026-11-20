from __future__ import annotations

import ufl

from fenics_constitutive.models.interfaces import StressStrainConstraint

__all__ = ["ufl_mandel", "ufl_mandel_strain", "ufl_tensor_from_mandel"]


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
        case StressStrainConstraint.UNIAXIAL_STRAIN | StressStrainConstraint.UNIAXIAL_STRESS:
            return ufl.nabla_grad(u)
        case (
            StressStrainConstraint.PLANE_STRAIN
            | StressStrainConstraint.PLANE_STRESS
            | StressStrainConstraint.FULL
        ):
            return ufl_mandel(ufl.nabla_grad(u))


def ufl_mandel(A: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    """
    Mandel vector of the symmetric part of a tensor. A 3x3 tensor gives
    $[A_{11}, A_{22}, A_{33}, \sqrt2 A_{12}, \sqrt2 A_{13}, \sqrt2 A_{23}]$, a 2x2 tensor gives the
    plane vector $[A_{11}, A_{22}, 0, \sqrt2 A_{12}]$. Off-diagonal entries are symmetrized.

    Args:
        A: A 2x2 or 3x3 UFL tensor.

    Returns:
        Vector-valued UFL expression in Mandel notation.
    """
    s = 1 / 2**0.5
    match A.ufl_shape:
        case (2, 2):
            return ufl.as_vector([A[0, 0], A[1, 1], 0.0, s * (A[0, 1] + A[1, 0])])
        case (3, 3):
            return ufl.as_vector(
                [
                    A[0, 0],
                    A[1, 1],
                    A[2, 2],
                    s * (A[0, 1] + A[1, 0]),
                    s * (A[0, 2] + A[2, 0]),
                    s * (A[1, 2] + A[2, 1]),
                ]
            )
        case shape:
            msg = f"Mandel notation is not defined for a tensor of shape {shape}."
            raise ValueError(msg)


def ufl_tensor_from_mandel(v: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    """
    Symmetric 3x3 tensor from its Mandel vector, the inverse of `ufl_mandel` for 3x3 tensors.

    Args:
        v: Vector-valued UFL expression with 6 components.

    Returns:
        3x3 UFL tensor.
    """
    if v.ufl_shape != (6,):
        msg = f"Expected a Mandel vector with 6 components, got shape {v.ufl_shape}."
        raise ValueError(msg)
    s = 1 / 2**0.5
    return ufl.as_matrix(
        [
            [v[0], s * v[3], s * v[4]],
            [s * v[3], v[1], s * v[5]],
            [s * v[4], s * v[5], v[2]],
        ]
    )
