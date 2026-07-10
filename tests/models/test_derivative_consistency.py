"""Finite-difference consistency checks of the analytical tangents for all
plasticity and damage models.

For local models the tangent is compared against central differences of the
stress increment map; for gradient-enhanced models all four partial tangents
(``dsigma_deps``, ``dsigma_dnonlocal``, ``dlocal_deps``, ``dlocal_dnonlocal``)
are compared against central differences in the Mandel strain and the nonlocal
quantity. Stress and history are reset before every perturbed call, so the
finite-difference result is exactly the consistent (algorithmic) derivative
that the analytical tangent claims to be.
"""

from __future__ import annotations

import numpy as np
import pytest

from fenics_constitutive.models.interfaces import (
    NonlocalTangents,
    StressStrainConstraint,
)
from fenics_constitutive.models.mises_plasticity_isotropic_hardening import VonMises3D
from fenics_constitutive.models.peerlings_gradient_damage import (
    PeerlingsGradientPerfectDamage,
)
from fenics_constitutive.models.rust_models import (
    DPHDamage3D,
    DPHWithoutDamage3D,
    DruckerPrager3D,
    DruckerPragerHyperbolic3D,
    Engelen3D,
    EngelenAnalytical3D,
    IsotropicMises3D,
    MisesPlasticityLinearHardening3D,
    PeerlingsGradientPerfectDamage3D,
)

SQ2 = np.sqrt(2.0)
FD_STEP = 1e-7
# central differences leave O(h^2 * curvature) noise; the strain-norm local
# quantity has curvature ~1/||eps||^2, so a few 1e-6 is the FD noise floor
RTOL = 5e-6

UNI = np.array([1.0, 0, 0, 0, 0, 0])
SHEAR = np.array([0, 0, 0, 1.0, 0.5, 0.2]) / np.linalg.norm([1.0, 0.5, 0.2])
RAND = 0.3 * np.random.default_rng(42).standard_normal(6)
MIX = (UNI + SHEAR + RAND) / np.linalg.norm(UNI + SHEAR + RAND)
CMIX = (-UNI + 0.5 * SHEAR) / np.linalg.norm(-UNI + 0.5 * SHEAR)

MISES_PARAMS = {
    "mu": np.array([80.77]),
    "kappa": np.array([166.67]),
    "y_0": np.array([0.2]),
    "h": np.array([10.0]),
}
DP_PARAMS = {
    "mu": np.array([80.77]),
    "kappa": np.array([166.67]),
    "a": np.array([0.15]),
    "b": np.array([0.05]),
    "b_flow": np.array([0.05]),
}
DPH_PARAMS = {
    "mu": np.array([80.77]),
    "kappa": np.array([166.67]),
    "a": np.array([0.15]),
    "b": np.array([0.05]),
    "d": np.array([0.1]),
    "h": np.array([5.0]),
    "b_flow": np.array([0.05]),
}
DPH_DAMAGE_PARAMS = {
    "mu": np.array([80.77]),
    "kappa": np.array([166.67]),
    "a_y": np.array([0.1]),
    "b_y": np.array([0.05]),
    "d_y": np.array([0.3]),
    "a_r": np.array([0.05]),
    "b_r": np.array([0.05]),
    "d_r": np.array([0.15]),
    "e_f": np.array([0.01]),
    "h": np.array([5.0]),
    "radial_factor": np.array([0.5]),
    "alpha_0": np.array([1e-4]),
    "omega_max": np.array([0.9]),
}
ENGELEN_PARAMS = {
    "mu": np.array([80.77]),
    "kappa": np.array([166.67]),
    "y_0": np.array([0.2]),
    "h": np.array([10.0]),
    "alpha_0": np.array([1e-4]),
    "e_f": np.array([0.01]),
    "omega_max": np.array([0.9]),
}
PEERLINGS_PARAMS = {
    "mu": np.array([80.77]),
    "kappa": np.array([166.67]),
    "eps_0": np.array([1e-4]),
    "omega_max": np.array([0.9]),
}


def mandel_to_grad(eps6: np.ndarray) -> np.ndarray:
    g = np.zeros((3, 3))
    g[0, 0], g[1, 1], g[2, 2] = eps6[0], eps6[1], eps6[2]
    g[0, 1] = g[1, 0] = eps6[3] / SQ2
    g[0, 2] = g[2, 0] = eps6[4] / SQ2
    g[1, 2] = g[2, 1] = eps6[5] / SQ2
    return g.flatten()


def make_history(law) -> dict[str, np.ndarray]:
    if law.history_dim is None:
        return {}
    return {
        key: np.zeros(np.prod(value)).flatten()
        for key, value in law.history_dim.items()
    }


def copy_history(history: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value.copy() for key, value in history.items()}


def assert_close(analytical, fd, label: str) -> None:
    scale = max(np.abs(fd).max(), np.abs(analytical).max(), 1e-12)
    err = np.abs(analytical - fd).max() / scale
    assert err < RTOL, f"{label}: rel. error {err:.3e}"


def check_local_tangent(law, steps, test_eps) -> bool:
    """Returns True if the test step was inelastic (history changed)."""
    stress = np.zeros(6)
    history = make_history(law)
    for eps6 in steps:
        law.evaluate(0.0, 0.0, mandel_to_grad(eps6), stress, np.zeros(36), history)
    s_base, h_base = stress.copy(), copy_history(history)

    def run(eps6):
        s = s_base.copy()
        h = copy_history(h_base)
        law.evaluate(0.0, 0.0, mandel_to_grad(eps6), s, None, h)
        return s

    tangent = np.zeros(36)
    s1, h1 = s_base.copy(), copy_history(h_base)
    law.evaluate(0.0, 0.0, mandel_to_grad(test_eps), s1, tangent, h1)

    fd = np.zeros((6, 6))
    for j in range(6):
        ep, em = test_eps.copy(), test_eps.copy()
        ep[j] += FD_STEP
        em[j] -= FD_STEP
        fd[:, j] = (run(ep) - run(em)) / (2 * FD_STEP)
    assert_close(tangent.reshape(6, 6), fd, "dsigma_deps")
    return any(np.abs(h1[k] - h_base[k]).max() > 1e-14 for k in h1)


def new_tangents() -> NonlocalTangents:
    return NonlocalTangents(
        dsigma_deps=np.zeros(36),
        dsigma_dnonlocal=np.zeros(6),
        dlocal_deps=np.zeros(6),
        dlocal_dnonlocal=np.zeros(1),
    )


def check_gradient_tangents(law, steps, test_eps, nonlocal_factor=1.0) -> bool:
    """Drives the model with lagged nonlocal coupling (nonlocal input = previous
    local output), then checks all four tangents at a test step whose nonlocal
    input is ``nonlocal_factor`` times the current local quantity.
    ``nonlocal_factor < 1`` probes the frozen-damage branch. Returns True if
    the test step was inelastic."""
    stress = np.zeros(6)
    local = np.zeros(1)
    history = make_history(law)
    for eps6 in steps:
        law.evaluate(
            0.0, 0.0, mandel_to_grad(eps6), local.copy(), stress, local,
            new_tangents(), history,
        )
    nl_val = nonlocal_factor * local[0]
    s_base, l_base, h_base = stress.copy(), local.copy(), copy_history(history)

    def run(eps6, nl):
        s, l, h = s_base.copy(), l_base.copy(), copy_history(h_base)
        law.evaluate(0.0, 0.0, mandel_to_grad(eps6), np.array([nl]), s, l, None, h)
        return s, l.copy()

    tg = new_tangents()
    s1, l1, h1 = s_base.copy(), l_base.copy(), copy_history(h_base)
    law.evaluate(
        0.0, 0.0, mandel_to_grad(test_eps), np.array([nl_val]), s1, l1, tg, h1
    )

    dsig_deps_fd = np.zeros((6, 6))
    dloc_deps_fd = np.zeros(6)
    for j in range(6):
        ep, em = test_eps.copy(), test_eps.copy()
        ep[j] += FD_STEP
        em[j] -= FD_STEP
        sp, lp = run(ep, nl_val)
        sm, lm = run(em, nl_val)
        dsig_deps_fd[:, j] = (sp - sm) / (2 * FD_STEP)
        dloc_deps_fd[j] = (lp[0] - lm[0]) / (2 * FD_STEP)
    sp, lp = run(test_eps, nl_val + FD_STEP)
    sm, lm = run(test_eps, nl_val - FD_STEP)

    assert_close(tg.dsigma_deps.reshape(6, 6), dsig_deps_fd, "dsigma_deps")
    assert_close(tg.dlocal_deps, dloc_deps_fd, "dlocal_deps")
    assert_close(tg.dsigma_dnonlocal, (sp - sm) / (2 * FD_STEP), "dsigma_dnonlocal")
    assert_close(
        tg.dlocal_dnonlocal[0], (lp[0] - lm[0]) / (2 * FD_STEP), "dlocal_dnonlocal"
    )
    return abs(l1[0] - l_base[0]) > 1e-14 or any(
        np.abs(h1[k] - h_base[k]).max() > 1e-14 for k in h1
    )


@pytest.mark.parametrize(
    "make_law",
    [
        lambda: IsotropicMises3D(MISES_PARAMS),
        lambda: MisesPlasticityLinearHardening3D(MISES_PARAMS),
        lambda: VonMises3D(
            {"p_ka": 166.67, "p_mu": 80.77, "p_y0": 0.2, "p_y00": 0.3, "p_w": 20.0}
        ),
    ],
    ids=["IsotropicMises3D", "MisesPlasticityLinearHardening3D", "VonMises3D-python"],
)
def test_mises_tangent(make_law):
    assert not check_local_tangent(make_law(), [], 5e-4 * MIX)
    assert check_local_tangent(make_law(), [2e-3 * MIX] * 3, 1e-3 * MIX)
    assert check_local_tangent(make_law(), [2e-3 * MIX] * 3, 1e-3 * SHEAR)


@pytest.mark.parametrize("b_flow", [0.05, 0.01], ids=["associated", "non-associated"])
def test_drucker_prager_tangent(b_flow):
    params = {**DP_PARAMS, "b_flow": np.array([b_flow])}
    assert not check_local_tangent(DruckerPrager3D(params), [], -2e-4 * UNI)
    assert check_local_tangent(DruckerPrager3D(params), [2e-3 * CMIX] * 3, 1e-3 * CMIX)


@pytest.mark.parametrize("b_flow", [0.05, 0.01], ids=["associated", "non-associated"])
def test_drucker_prager_hyperbolic_tangent(b_flow):
    params = {**DPH_PARAMS, "b_flow": np.array([b_flow])}
    assert not check_local_tangent(DruckerPragerHyperbolic3D(params), [], -2e-4 * UNI)
    assert check_local_tangent(
        DruckerPragerHyperbolic3D(params), [2e-3 * CMIX] * 3, 1e-3 * CMIX
    )


def test_dph_without_damage_tangent():
    assert not check_local_tangent(DPHWithoutDamage3D(DPH_DAMAGE_PARAMS), [], -2e-4 * UNI)
    assert check_local_tangent(
        DPHWithoutDamage3D(DPH_DAMAGE_PARAMS), [1e-3 * MIX] * 3, 5e-4 * MIX
    )


def test_dph_damage_tangents():
    assert not check_gradient_tangents(DPHDamage3D(DPH_DAMAGE_PARAMS), [], 2e-4 * MIX)
    # growing damage
    assert check_gradient_tangents(
        DPHDamage3D(DPH_DAMAGE_PARAMS), [1e-3 * MIX] * 4, 5e-4 * MIX
    )
    # frozen damage: nonlocal input strictly below the stored maximum
    assert check_gradient_tangents(
        DPHDamage3D(DPH_DAMAGE_PARAMS), [1e-3 * MIX] * 4, 5e-4 * MIX, nonlocal_factor=0.3
    )


@pytest.mark.parametrize(
    "cls", [Engelen3D, EngelenAnalytical3D], ids=["Engelen3D", "EngelenAnalytical3D"]
)
def test_engelen_tangents(cls):
    assert not check_gradient_tangents(cls(ENGELEN_PARAMS), [], 5e-4 * MIX)
    assert check_gradient_tangents(cls(ENGELEN_PARAMS), [2e-3 * MIX] * 4, 1e-3 * MIX)
    assert check_gradient_tangents(
        cls(ENGELEN_PARAMS), [2e-3 * MIX] * 4, 1e-3 * MIX, nonlocal_factor=0.3
    )


def test_engelen_analytical_matches_newton():
    """The closed-form radial return must reproduce the general Newton return
    mapping (same yield function) to solver accuracy."""
    laws = [Engelen3D(ENGELEN_PARAMS), EngelenAnalytical3D(ENGELEN_PARAMS)]
    states = [
        {"stress": np.zeros(6), "local": np.zeros(1), "history": make_history(law)}
        for law in laws
    ]
    for _ in range(8):
        for law, st in zip(laws, states):
            law.evaluate(
                0.0, 0.0, mandel_to_grad(2e-3 * MIX), st["local"].copy(),
                st["stress"], st["local"], None, st["history"],
            )
        assert np.allclose(states[0]["stress"], states[1]["stress"], atol=1e-7)
        assert np.allclose(states[0]["local"], states[1]["local"], atol=1e-9)
    assert states[0]["local"][0] > 0.0, "plasticity was not reached in test"


@pytest.mark.parametrize(
    "make_law",
    [
        lambda: PeerlingsGradientPerfectDamage3D(PEERLINGS_PARAMS),
        lambda: PeerlingsGradientPerfectDamage(
            {"E": 210.0, "nu": 0.3, "eps_0": 1e-4, "omega_max": 0.9},
            StressStrainConstraint.FULL,
        ),
    ],
    ids=["Peerlings3D-rust", "Peerlings-python"],
)
def test_peerlings_tangents(make_law):
    check_gradient_tangents(make_law(), [], 5e-5 * MIX)  # below eps_0
    assert check_gradient_tangents(make_law(), [5e-4 * MIX] * 2, 2e-4 * MIX)
    assert check_gradient_tangents(
        make_law(), [5e-4 * MIX] * 2, 2e-4 * MIX, nonlocal_factor=0.3
    )
