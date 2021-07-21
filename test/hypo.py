 
import warnings

import dolfin as df
import numpy as np
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
from scipy.linalg import eigvals
import constitutive as c


E, nu  = 42.0, 0.3
rho = 1e-1
t_end = 300
n=300
h=0.5

law = c.Hypoelasticity(E,nu)
stress = np.random.random(6*n)
L = np.random.random(9*n)
t = np.ones(n)

ip_loop = c.IpLoop()
ip_loop.add_law(law)
ip_loop.resize(n)

ip_loop.set(c.Q.SIGMA, stress)
ip_loop.set(c.Q.L, L)
ip_loop.set(c.Q.TIME_STEP, np.ones_like(t) * h)
ip_loop.evaluate()
assert len(ip_loop.get(c.Q.SIGMA)) == len(stress)
ip_loop.set(c.Q.SIGMA, stress)
ip_loop.set(c.Q.L, L)
ip_loop.set(c.Q.TIME_STEP, np.ones_like(t) * h)
ip_loop.evaluate()
assert len(ip_loop.get(c.Q.SIGMA)) == len(stress)
ip_loop.set(c.Q.SIGMA, stress)
ip_loop.set(c.Q.L, L)
ip_loop.set(c.Q.TIME_STEP, np.ones_like(t) * h)
ip_loop.evaluate()
assert len(ip_loop.get(c.Q.SIGMA)) == len(stress)
