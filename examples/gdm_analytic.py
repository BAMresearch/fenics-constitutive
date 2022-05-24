"""

Comparison with the analytic solution
*************************************

The Peerlings paper cited above includes a discussion on an analytic solution 
of the model for the ``damage_perfect`` damage law. A 1D bar of length $L$
(here modeled as a thin 2D structure) has a cross section reduction $\alpha$ 
over the length of $W$ and is loaded by a displacement BC $\Delta L$.

Three regions form: 
    * damage in the weakened cross section
    * damage in the unweakened cross section
    * no damage

and the authors provide a solution of the PDE system for each of the regions.
Finding the remaining integration constants is left to the reader. Here, it
is solved using ``sympy``. We also subclass from ``dolfin.UserExpression`` to
interpolate the analytic solution into function spaces or calculate error norms.
"""

import numpy as np


class PeerlingsAnalytic:
    def __init__(self):
        self.L, self.W, self.deltaL, self.alpha = 100.0, 10.0, 0.05, 0.1
        self.E, self.kappa0, self.l = 20000.0, 1.0e-4, 1.0

        self._calculate_coeffs()

    def _calculate_coeffs(self):
        """
        The analytic solution is following Peerlings paper (1996) but with
        b(paper) = b^2 (here)
        g(paper) = g^2 (here)
        c(paper) = l^2 (here)
        This modification eliminates all the sqrts in the formulations.
        Plus: the formulation of the GDM in terms of l ( = sqrt(c) ) is
        more common in modern publications.
        """

        # imports only used here...
        from sympy import Symbol, symbols, N, integrate, cos, exp, lambdify
        import scipy.optimize

        # unknowns
        x = Symbol("x")
        unknowns = symbols("A1, A2, B1, B2, C, b, g, w")
        A1, A2, B1, B2, C, b, g, w = unknowns

        l = self.l
        kappa0 = self.kappa0

        # 0 <= x <= W/2
        e1 = C * cos(g / l * x)
        # W/2 <  x <= w/2
        e2 = B1 * exp(b / l * x) + B2 * exp(-b / l * x)
        # w/2 <  x <= L/2
        e3 = A1 * exp(x / l) + A2 * exp(-x / l) + (1 - b * b) * kappa0

        de1, de2, de3 = e1.diff(x), e2.diff(x), e3.diff(x)

        eq1 = N(e1.subs(x, self.W / 2) - e2.subs(x, self.W / 2))
        eq2 = N(de1.subs(x, self.W / 2) - de2.subs(x, self.W / 2))
        eq3 = N(e2.subs(x, w / 2) - kappa0)
        eq4 = N(de2.subs(x, w / 2) - de3.subs(x, w / 2))
        eq5 = N(e3.subs(x, w / 2) - kappa0)
        eq6 = N(de3.subs(x, self.L / 2))
        eq7 = N((1 - self.alpha) * (1 + g * g) - (1 - b * b))
        eq8 = N(
            integrate(e1, (x, 0, self.W / 2))
            + integrate(e2, (x, self.W / 2, w / 2))
            + integrate(e3, (x, w / 2, self.L / 2))
            - self.deltaL / 2
        )

        eqs = [
            lambdify(unknowns, eq) for eq in [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8]
        ]

        def global_func(x):
            return np.array([eqs[i](*x) for i in range(8)])

        result = scipy.optimize.root(
            global_func, [0.0, 5e2, 3e-7, 7e-3, 3e-3, 3e-1, 2e-1, 4e1]
        )
        if not result["success"]:
            raise RuntimeError(
                "Could not find the correct coefficients. Try to tweak the initial values."
            )

        self.coeffs = result["x"]

    def e(self, x):
        A1, A2, B1, B2, C, b, g, w = self.coeffs
        if x <= self.W / 2.0:
            return C * np.cos(g / self.l * x)
        elif x <= w / 2.0:
            return B1 * np.exp(b / self.l * x) + B2 * np.exp(-b / self.l * x)
        else:
            return (
                (1.0 - b * b) * self.kappa0
                + A1 * np.exp(x / self.l)
                + A2 * np.exp(-x / self.l)
            )
