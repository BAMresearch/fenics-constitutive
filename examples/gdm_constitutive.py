"""
Gradient damage constitutive law
--------------------------------
"""

import numpy as np
import ufl
from dataclasses import dataclass

"""
Damage law
**********

The simplest case that is used in Peerlings et al. for an analytic solution is
the perfect damage model that, if inserted in the stress-strain relationship, 
looks like
::

    stress
 ft |   _______________
    |  /
    | /:
    |/ :
    0--:--------------> strain
       k0 
"""


def damage_perfect(mat, kappa):
    k0 = mat.ft / mat.E
    return 1.0 - k0 / kappa, k0 / kappa ** 2


"""
For the characteristic strain softening, an exponential damage law is commonly
used. After reaching the peak load - the tensile strength $f_t$, the curve
shows an exponential drop to the residual stress $(1-\alpha)f_t$.
::

    stress
 ft |   
    |  /\
    | /: ` .
    |/ :     ` .. _____ (1-alpha)*ft
    0--:--------------> strain
       k0 
"""


def damage_exponential(mat, k):
    k0 = mat.ft / mat.E
    a = mat.alpha
    b = mat.beta

    w = 1.0 - k0 / k * (1.0 - a + a * np.exp(b * (k0 - k)))
    dw = k0 / k * ((1.0 / k + b) * a * np.exp(b * (k0 - k)) + (1.0 - a) / k)

    return w, dw


"""
Constitutive law
****************

Basically `Hooke's law in plane strain <https://en.wikipedia.org/wiki/Hooke%27s_law#Plane_strain>`_ with the factor $(1-\omega)$. 

.. math ::
    \bm \sigma &= (1 - \omega(\kappa)) \bm C : \bm \varepsilon \\
    \frac{\partial \bm \sigma}{\partial \bm \varepsilon} &= (1 - \omega) \bm C  \\
    \frac{\partial \bm \sigma}{\partial \bar \varepsilon} &= - \omega \bm C : \bm \varepsilon \frac{\mathrm d \omega}{\mathrm d\kappa}\frac{\mathrm d\kappa}{\mathrm d \bar \varepsilon}

"""


def hooke(mat, eps, kappa, dkappa_de):
    """
    mat:
        material parameters
    eps:
        vector of Nx3 where each of the N rows is a 2D strain in Voigt notation
    kappa:
        current value of the history variable kappa
    dkappa_de:
        derivative of kappa w.r.t the nonlocal equivalent strains e
    """
    E, nu = mat.E, mat.nu
    l = E * nu / (1 + nu) / (1 - 2 * nu)
    m = E / (2.0 * (1 + nu))
    C = np.array([[2 * m + l, l, 0], [l, 2 * m + l, 0], [0, 0, m]])

    w, dw = mat.dmg(mat, kappa)

    sigma = eps @ C * (1 - w)[:, None]
    dsigma_deps = np.tile(C.flatten(), (len(kappa), 1)) * (1 - w)[:, None]

    dsigma_de = -eps @ C * dw[:, None] * dkappa_de[:, None]

    return sigma, dsigma_deps, dsigma_de


"""
Strain norm
***********

The local equivalent strain $\| \bm \varepsilon \|$ is defined as

.. math::
  \|\bm \varepsilon\| = \frac{k-1}{2k(1-2\nu)}I_1 + \frac{1}{2k}\sqrt{\left(\frac{k-1}{1-2\nu}I_1\right)^2 + \frac{2k}{(1+\nu)^2}J_2}.

$I_1$ is the first strain invariant and $J_2$ is the second deviatoric strain invariant. 
The parameter $k=f_c/f_t$ controls different material responses in compression (compressive strength $f_c$) and tension (tensile strength $f_t$).

See: `Comparison of nonlocal approaches in continuum damage mechanics, de Vree et al., 1995 <http://dx.doi.org/10.1016/0045-7949(94)00501-S>`_

Note that the implementation here is only valid for 2D plane strain!
"""


def modified_mises_strain_norm(mat, eps):
    nu, k = mat.nu, mat.k

    K1 = (k - 1.0) / (2.0 * k * (1.0 - 2.0 * nu))
    K2 = 3.0 / (k * (1.0 + nu) ** 2)

    exx, eyy, exy = eps[0::3], eps[1::3], eps[2::3]
    I1 = exx + eyy
    J2 = 1.0 / 6.0 * ((exx - eyy) ** 2 + exx ** 2 + eyy ** 2) + (0.5 * exy) ** 2

    A = np.sqrt(K1 ** 2 * I1 ** 2 + K2 * J2) + 1.0e-14
    eeq = K1 * I1 + A

    dJ2dexx = 1.0 / 3.0 * (2 * exx - eyy)
    dJ2deyy = 1.0 / 3.0 * (2 * eyy - exx)
    dJ2dexy = 0.5 * exy

    deeq = np.empty_like(eps)
    deeq[0::3] = K1 + 1.0 / (2 * A) * (2 * K1 * K1 * I1 + K2 * dJ2dexx)
    deeq[1::3] = K1 + 1.0 / (2 * A) * (2 * K1 * K1 * I1 + K2 * dJ2deyy)
    deeq[2::3] = 1.0 / (2 * A) * (K2 * dJ2dexy)
    return eeq, deeq


"""
Complete constitutive class
***************************

* contains the material parameters
* stores the values of all integration points
* calculates those fields for given $\bm \varepsilon, \bar \varepsilon$

"""


@dataclass
class GDMPlaneStrain:
    # Young's modulus          [N/mm²]
    E: float = 20000.0
    # Poisson's ratio            [-]
    nu: float = 0.2
    # nonlocal length parameter [mm]
    l: float = 200 ** 0.5
    # tensile strength          [N/mm²]
    ft: float = 2.0
    # compressive-tensile ratio   [-]
    k: float = 10.0
    # residual strength factor   [-]
    alpha: float = 0.99
    # fracture energy parameters [-]
    beta: float = 100.0
    # history variable           [-]
    kappa: None = None
    # damage law
    dmg: None = damage_exponential

    def eps(self, v):
        e = ufl.sym(ufl.grad(v))
        return ufl.as_vector([e[0, 0], e[1, 1], 2 * e[0, 1]])

    def kappa_kkt(self, e):
        if self.kappa is None:
            self.kappa = self.ft / self.E
        return np.maximum(e, self.kappa)

    def evaluate(self, eps_flat, e):
        kappa = self.kappa_kkt(e)
        dkappa_de = (e >= kappa).astype(int)

        eps = eps_flat.reshape(-1, 3)
        self.sigma, self.dsigma_deps, self.dsigma_de = hooke(
            self, eps, kappa, dkappa_de
        )
        self.eeq, self.deeq = modified_mises_strain_norm(self, eps_flat)

    def update(self, e):
        self.kappa = self.kappa_kkt(e)
