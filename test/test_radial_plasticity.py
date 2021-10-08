import unittest
import dolfin as df
import numpy as np
import constitutive as c
np.set_printoptions(precision=4)
"""
######################################################
##             SJARD PLASTICITY MODEL BEGIN
######################################################
"""

dev = np.array(
    [[2./3., -1./3., -1./3., 0., 0., 0.],
        [-1./3., 2./3., -1./3., 0., 0., 0.],
        [-1./3., -1./3., 2./3., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 1.]])

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
        self.σn = np.zeros(6)
        self.pn = 0.0
        self.εn = np.zeros(6)

        self.σn1 = np.zeros(6)
        self.pn1 = 0.0
        self.εn1 = np.zeros(6)

    def evaluate(self, ε):
        Δε = ε - self.εn
        self.εn1 = ε

        σ_tr = self.σn + self.Ce @ Δε
        σ_dev = self.dev @ σ_tr

        σ_eq = np.sqrt((3 / 2) * np.inner(σ_dev, σ_dev))

        f = σ_eq - self.sig0 - self.H * self.pn

        if f <= 0:
            # elastic strain
            self.σn1 = σ_tr
            self.pn1 = self.pn
        else:
            # stress return
            Δp = f / (3 * self.mu + self.H)
            self.pn1 = self.pn + Δp
            β = (3 * self.mu * Δp) / σ_eq
            self.σn1 = σ_tr - β * σ_dev

    def update(self):
        self.σn = self.σn1
        self.pn = self.pn1
        self.εn = self.εn1

class RadialMisesTestIntegrationPoint(unittest.TestCase):
    def test_sigma_no_hardening(self):
        E = 420.
        nu = 0.3
        sig0 = 2
        H = 0


        mises_analytical = c.AnalyticMisesPlasticity(E,nu,sig0,H, total_strains=True, tangent=False)
        mises_python = VonMisesAnalytical(E, nu, sig0, H)
        mises_radial = c.RadialReturnMisesPlasticity(E,nu,sig0,H)
        mises_Y = c.RadialMisesYieldSurface(sig0, H)
        mises_radial_2 = c.RadialReturnPlasticity(E,nu,mises_Y)
        #np.testing.assert_allclose(mises_python.Ce, mises_analytical.C)

        loop_analytical = c.IpLoop()
        loop_radial = c.IpLoop()
        loop_radial_2 = c.IpLoop()

        loop_analytical.add_law(mises_analytical, np.array([0]))
        loop_analytical.resize(1)
        loop_radial.add_law(mises_radial, np.array([0]))
        loop_radial.resize(1)
        loop_radial_2.add_law(mises_radial_2, np.array([0]))
        loop_radial_2.resize(1)
        eps = np.array([1.,0.,0.,0.,0.,0.])
        eps = eps /np.linalg.norm(eps)
        s = np.linspace(0,0.1,10)
        eps_eq = np.zeros_like(s)
        sig_eq = np.zeros_like(s)
        for si in s:
            #print(si)
            loop_analytical.set(c.Q.EPS, eps*si)
            loop_analytical.evaluate()
            loop_radial.set(c.Q.EPS, eps*si)
            loop_radial.evaluate()
            loop_radial_2.set(c.Q.EPS, eps*si)
            loop_radial_2.evaluate()
            mises_python.evaluate(eps*si)
            np.testing.assert_array_almost_equal(loop_analytical.get(c.Q.SIGMA), loop_radial.get(c.Q.SIGMA),decimal = 10)
            np.testing.assert_array_almost_equal(loop_analytical.get(c.Q.SIGMA), loop_radial_2.get(c.Q.SIGMA),decimal = 10)
            np.testing.assert_array_almost_equal(loop_analytical.get(c.Q.SIGMA), mises_python.σn1,decimal = 10)

            loop_analytical.update()
            loop_radial.update()
            loop_radial_2.update()
            mises_python.update()

            #print(np.linalg.norm(loop.get(c.Q.SIGMA)- mises_analytical.σn1)/np.linalg.norm(loop.get(c.Q.SIGMA)))
            #print(loop_analytical.get(c.Q.SIGMA))
            #print(loop_radial.get(c.Q.SIGMA))
            #print(mises_python.σn1)

    def test_sigma_isotropic_hardening(self):
        E = 420.
        nu = 0.3
        sig0 = 2
        H = 0


        mises_analytical = c.AnalyticMisesPlasticity(E,nu,sig0,H, total_strains=True, tangent=False)
        mises_python = VonMisesAnalytical(E, nu, sig0, H)
        mises_radial = c.RadialReturnMisesPlasticity(E,nu,sig0,H)
        mises_Y = c.RadialMisesYieldSurface(sig0, H)
        mises_radial_2 = c.RadialReturnPlasticity(E,nu,mises_Y)

        loop_analytical = c.IpLoop()
        loop_radial = c.IpLoop()
        loop_radial_2 = c.IpLoop()

        loop_analytical.add_law(mises_analytical, np.array([0]))
        loop_analytical.resize(1)
        loop_radial.add_law(mises_radial, np.array([0]))
        loop_radial.resize(1)
        loop_radial_2.add_law(mises_radial_2, np.array([0]))
        loop_radial_2.resize(1)
        eps = np.array([1.,0.,0.,0.,0.,0.])
        eps = eps /np.linalg.norm(eps)
        s = np.linspace(0,0.1,10)
        eps_eq = np.zeros_like(s)
        sig_eq = np.zeros_like(s)
        for si in s:
            #print(si)
            loop_analytical.set(c.Q.EPS, eps*si)
            loop_analytical.evaluate()
            loop_radial.set(c.Q.EPS, eps*si)
            loop_radial.evaluate()
            loop_radial_2.set(c.Q.EPS, eps*si)
            loop_radial_2.evaluate()
            mises_python.evaluate(eps*si)
            np.testing.assert_array_almost_equal(loop_analytical.get(c.Q.SIGMA), loop_radial.get(c.Q.SIGMA),decimal = 10)
            np.testing.assert_array_almost_equal(loop_analytical.get(c.Q.SIGMA), loop_radial_2.get(c.Q.SIGMA),decimal = 10)
            np.testing.assert_array_almost_equal(loop_analytical.get(c.Q.SIGMA), mises_python.σn1,decimal = 10)

            loop_analytical.update()
            loop_radial.update()
            loop_radial_2.update()
            mises_python.update()

            # print("_______________________________")
            # print(loop_analytical.get(c.Q.SIGMA))
            # print(loop_radial.get(c.Q.SIGMA))
            # print(loop_radial_2.get(c.Q.SIGMA))
            # print(mises_analytical.get_internal_var(c.Q.LAMBDA), mises_radial.get_internal_var(c.Q.LAMBDA), mises_radial_2.get_internal_var(c.Q.LAMBDA))
if __name__ == "__main__":
    unittest.main()
