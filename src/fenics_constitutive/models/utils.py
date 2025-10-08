from __future__ import annotations


def lame_parameters(E: float, nu: float) -> tuple[float, float]:
    """Compute Lame parameters (mu, lam) from Young's modulus and Poisson's ratio."""
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam
