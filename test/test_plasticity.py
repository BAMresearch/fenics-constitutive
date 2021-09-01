import unittest
import dolfin as df
import numpy as np
from fenics_helpers import boundary
from fenics_helpers.timestepping import TimeStepper
import constitutive as c


def show_loading(loading, t0=0.0, t1=1.0, N=1000):
    import matplotlib.pyplot as plt

    ts = np.linspace(t0, t1, N)
    bcs = []
    for t in ts:
        bcs.append(loading(t))
    plt.plot(ts, bcs)
    plt.show()

"""
######################################################
##             SJARD PLASTICITY MODEL BEGIN
######################################################
"""

"""
######################################################
##             ABBAS PLASTICITY MODEL END
######################################################
"""



class TestPlasticity(unittest.TestCase):
    def test_integration_point(self):
        E = 42.
        nu = 0.3

        law = c.HookesLaw(E, nu, False, False)
        C = law.C

        mises = c.MisesYieldFunction(10, 5)
        hardening = c.StrainHardening()
        loop = c.IpLoop()


        plasticity = c.IsotropicHardeningPlasticity(C, mises, hardening, total_strains=True, tangent=False)
        loop.add_law(plasticity, np.array([0]))
        loop.resize(1)
        loop.set(c.Q.EPS, np.zeros(6))
        print("whe are mha")
        loop.evaluate()

        print(loop.get(cpp.Q.SIGMA))
        print("this is going well")


if __name__ == "__main__":
    unittest.main()
