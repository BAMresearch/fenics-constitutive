import unittest
import dolfin as df
import numpy as np
import constitutive as c

"""
######################################################
##             SJARD PLASTICITY MODEL BEGIN
######################################################
"""

def mises_norm(x):
    return np.sqrt()
class VonMisesAnalytical:
    def __init__(self,  E, nu,  sig0, H):
        self.lam = E * nu / (1 + nu) / (1 - 2 * nu)
        self.mu = E / (2.0 * (1 + nu))
        self.H = H
        self.sig0 = sig0
        self.Ce = np.array([[2*self.mu+self.lam, self.lam, self.lam, 0., 0., 0.],
             [self.lam, 2*self.mu+self.lam, self.lam, 0., 0., 0.],
             [self.lam, self.lam, 2*self.mu+self.lam, 0., 0., 0.],
             [0., 0., 0., 2*self.mu, 0., 0.],
             [0., 0., 0., 0., 2*self.mu, 0.],
             [0., 0., 0., 0., 0., 2*self.mu]])

        self.dev = np.array(
            [[2./3., -1./3., -1./3., 0., 0., 0.],
                [-1./3., 2./3., -1./3., 0., 0., 0.],
                [-1./3., -1./3., 2./3., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0.], 
                [0., 0., 0., 0., 1., 0.], 
                [0., 0., 0., 0., 0., 1.]])
        # Initialize state variables
        self.Ct = self.Ce
        self.σn = np.zeros(6)
        self.pn = 0.0
        self.εn = np.zeros(6)

        self.σn1 = np.zeros(6)
        self.pn1 = 0.0
        self.εn1 = np.zeros(6)

    def evaluate(self, ε):
        """
        
        Parameters
        ----------
        Δε : ndarray
            Array containing the strain-increment
        Returns
        -------
        """
        Δε = ε - self.εn
        self.εn1 = ε

        σ_tr = self.σn + self.Ce @ Δε
        σ_dev = self.dev @ σ_tr

        σ_eq = np.sqrt((3 / 2) * np.sum(σ_dev ** 2))

        f = σ_eq - self.sig0 - self.H * self.pn

        if f <= 0:
            # elastic strain
            self.Ct = self.Ce
            self.σn1 = σ_tr
            self.pn1 = self.pn
        else:
            # stress return
            Δp = f / (3 * self.mu + self.H)
            self.pn1 = self.pn + Δp
            n_elas = σ_dev / σ_eq
            β = (3 * self.mu * Δp) / σ_eq
            self.σn1 = σ_tr - β * σ_dev
            nxn = np.outer(n_elas, n_elas)
            self.Ct = (
                self.Ce
                - 3 * self.mu * (3 * self.mu / (3 * self.mu + self.H) - β) * nxn
                - 2 * self.mu * β * self.dev
            )

    def update(self):
        self.σn = self.σn1
        self.pn = self.pn1
        self.εn = self.εn1

class MisesTest(unittest.TestCase):
    def test_total_strains_no_tangent(self):
        E = 42.
        nu = 0.3
        sig0 = 2
        H = 4

        law = c.HookesLaw(E, nu, False, False)
        mises_analytical = VonMisesAnalytical(E, nu, sig0, H)

        mises_function = c.MisesYieldFunction(sig0, H)
        hardening = c.StrainHardening()
        loop = c.IpLoop()


        mises_newton = c.IsotropicHardeningPlasticity(law.C, mises_function, hardening, total_strains=True, tangent=False)
        
        loop.add_law(mises_newton, np.array([0]))
        loop.resize(1)
        eps = np.random.random(6)
        eps = eps /np.linalg.norm(eps)
        s = np.linspace(0,0.1,42)
        eps_eq = np.zeros_like(s)
        sig_eq = np.zeros_like(s)
        for si in s:
            loop.set(c.Q.EPS, eps*si)
            loop.evaluate()
            mises_analytical.evaluate(eps*si)
            np.testing.assert_allclose(loop.get(c.Q.SIGMA), mises_analytical.σn1)
            mises_analytical.update()
            loop.update()


if __name__ == "__main__":
    unittest.main()
