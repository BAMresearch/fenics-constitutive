#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 2021

@author: ajafari
"""

import constitutive as c
import numpy as np

# class RateIndependentHistory_PlasticWork(c.RateIndependentHistory):
#     def __init__(self, yield_function):
#         self.yf = yield_function
#         def p_and_diffs(sigma, kappa):
#             _, m, dm, _, mk = self.yf(sigma, kappa)
#             p = np.array([[np.dot(sigma, m)]])
#             dp_dsig = (m + dm.T @ sigma).reshape((1, -1))
#             dp_dk = np.array([[np.dot(sigma, mk)]])
#             return p, dp_dsig, dp_dk
#         self.p_and_diffs = p_and_diffs
#     def __call__(self, sigma, kappa):
#         return self.p_and_diffs(sigma, kappa)

class PlasticConsitutivePerfect(c.LinearElastic):
    def __init__(self, E, nu, constraint, yf):
        """
        yf: a callable with:
            argument:      stress state
            return values: f: yield surface evaluation
                            m: flow vector; e.g. with associated flow rule: the first derivative of yield surface w.r.t. stress
                            dm: derivative of flow vector w.r.t. stress; e.g. with associated flow rule: second derivative of yield surface w.r.t. stress
        """
        super().__init__(E, nu, constraint)
        self.yf = yf
        assert(self.ss_dim == self.yf.ss_dim)
    
    def correct_stress(self, sig_tr, k0=0, _Ct=True, tol=1e-9, max_iters=20):
        """
        sig_tr: trial (predicted) stress
        k0: last converged internal variable(s), which is not relevant here, but is given just for the sake of generality
        returns:
            sig_c: corrected_stress
            dl: lamda_dot
            Ct: corrected stiffness matrix
            m: flow vector (e.g. normal vector to the yield surface, if self.yf is associative)
        """
        
        ### WAY 1 ### Using Jacobian matrix to solve residuals (of the return mapping algorithm) in a coupled sense
        f, m, dm, _, _ = self.yf(sig_tr)
        if f > 0: # do return mapping
            Ct = None
            _d = len(sig_tr)
            # initial values of unknowns (we add 0.0 to something to have a copy regardless of type (float or np.array))
            sig_c = sig_tr + 0.0
            dl = 0.0
            # compute residuals of return-mapping (backward Euler)
            d_eps_p = dl * m # change in plastic strain
            es = sig_c - sig_tr + self.D @ d_eps_p
            ef = f
            e_norm = np.linalg.norm(np.append(es, ef))
            _it = 0
            while e_norm > tol and _it<=max_iters:
                A1 = np.append( np.eye(_d) + dl * self.D @ dm, (self.D @ m).reshape((-1,1)), axis=1 )
                A2 = np.append(m, 0).reshape((1,-1))
                Jac = np.append(A1, A2, axis=0)
                dx = np.linalg.solve(Jac, np.append(es, ef))
                sig_c -= dx[0:_d]     
                dl -= dx[_d:]
                f, m, dm, _, _ = self.yf(sig_c)
                d_eps_p = dl * m # change in plastic strain
                es = sig_c - sig_tr + self.D @ d_eps_p
                ef = f
                e_norm = np.linalg.norm(np.append(es, ef))
                _it += 1
            # after converging return-mapping:
            if _Ct:
                A1 = np.append( np.eye(_d) + dl * self.D @ dm, (self.D @ m).reshape((-1,1)), axis=1 )
                A2 = np.append(m, 0).reshape((1,-1))
                Jac = np.append(A1, A2, axis=0)
                inv_Jac = np.linalg.inv(Jac)
                Ct = inv_Jac[np.ix_(range(_d), range(_d))] @ self.D
            return sig_c, Ct, k0, d_eps_p
        else: # still elastic zone
            return sig_tr, self.D, k0, 0.0        
    
class PlasticConsitutiveRateIndependentHistory(PlasticConsitutivePerfect):
    def __init__(self, E, nu, constraint, yf, ri):
        """
        ri: an instance of RateIndependentHistory representing evolution of history variables
        , which is based on:
            kappa_dot = lamda_dot * p(sigma, kappa), where "p" is a plastic modulus function
        """
        super().__init__(E, nu, constraint, yf)
        assert(isinstance(ri, c.RateIndependentHistory))
        self.ri = ri # ri: rate-independent
    
    def correct_stress(self, sig_tr, k0=0.0, tol=1e-9, max_iters=20, _Ct=True):
        """
        overwritten to the superclass'
        one additional equation to be satisfied is the rate-independent equation:
            kappa_dot = lamda_dot * self.ri.p
        , for the evolution of the history variable(s) k
        """
        
        ### Solve residuals (of the return mapping algorithm) equal to ZERO, in a coupled sense (using Jacobian matrix based on backward Euler)
        f, m, dm, fk, _ = self.yf(sig_tr, k0)
        if f > 0: # do return mapping
            Ct = None
            _d = len(sig_tr)
            # initial values of unknowns (we add 0.0 to something to have a copy regardless of type (float or np.array))
            sig_c = sig_tr + 0.0
            k = k0 + 0.0
            dl = 0.0
            # compute residuals of return-mapping (backward Euler)
            d_eps_p = dl * m # change in plastic strain
            es = sig_c - sig_tr + self.D @ d_eps_p
            p, dp_dsig, dp_dk = self.ri(sig_c, k)
            p = np.atleast_2d(p); dp_dk = np.atleast_2d(dp_dk); dp_dsig=np.atleast_2d(dp_dsig);
            if max(dp_dsig.shape) != _d:
                dp_dsig = dp_dsig * np.ones((1, _d))
            ek = k - k0 - dl * p
            ef = f
            e_norm = np.linalg.norm(np.append(np.append(es, ek), ef))
            _it = 0
            while e_norm > tol and _it<=max_iters:
                A1 = np.append( np.append(np.eye(_d) + dl * self.D @ dm, np.zeros((_d,1)), axis=1) \
                                , (self.D @ m).reshape((-1,1)), axis=1 )
                A2 = np.append(np.append(- dl * dp_dsig, 1 - dl * dp_dk, axis=1), -p, axis=1)
                A3 = np.append(np.append(m, fk), 0).reshape((1,-1))
                Jac = np.append(np.append(A1, A2, axis=0), A3, axis=0)
                dx = np.linalg.solve(Jac, np.append(np.append(es, ek), ef))
                sig_c -= dx[0:_d]     
                k -= dx[_d:_d+1]
                dl -= dx[_d+1:]
                f, m, dm, fk, _ = self.yf(sig_c, k)
                d_eps_p = dl * m # change in plastic strain
                es = sig_c - sig_tr + self.D @ d_eps_p
                p, dp_dsig, dp_dk = self.ri(sig_c, k)
                p = np.atleast_2d(p); dp_dk = np.atleast_2d(dp_dk); dp_dsig=np.atleast_2d(dp_dsig);
                if max(dp_dsig.shape) != _d:
                    dp_dsig = np.zeros((1, _d))
                ek = k - k0 - dl * p
                ef = f
                e_norm = np.linalg.norm(np.append(np.append(es, ek), ef))
                _it += 1
            # after converging return-mapping:
            if _Ct:
                A1 = np.append( np.append(np.eye(_d) + dl * self.D @ dm, np.zeros((_d,1)), axis=1) \
                                , (self.D @ m).reshape((-1,1)), axis=1 )
                A2 = np.append(np.append(- dl * dp_dsig, 1 - dl * dp_dk, axis=1), -p, axis=1)
                A3 = np.append(np.append(m, fk), 0).reshape((1,-1))
                Jac = np.append(np.append(A1, A2, axis=0), A3, axis=0)
                inv_Jac = np.linalg.inv(Jac)
                Ct = inv_Jac[np.ix_(range(_d), range(_d))] @ self.D
            return sig_c, Ct, k, d_eps_p
        else: # still elastic zone
            return sig_tr, self.D, k0, 0.0
        
class PlasticityIPLoopInPython:
    def __init__(self, mat):
        self.mat = mat
        self.q = mat.ss_dim

    def add_law(self, _):
        pass # just to match the c++ interface

    def resize(self, n):  
        self.n = n
        q = 3
        self.stress = np.zeros((self.n, q))
        self.dstress = np.zeros((self.n,  q * q))
        self.eps_p = np.zeros((self.n, q))
        self.kappa1 = np.zeros(self.n)
        self.kappa = np.zeros(self.n)
        self.deps_p = np.zeros((self.n, q))

    def get(self, what):
        if what == c.Q.SIGMA:
            return self.stress
        if what == c.Q.DSIGMA_DEPS:
            return self.dstress

    def update(self, all_strains):
        self.kappa[:] = self.kappa1[:]
        self.eps_p += self.deps_p

    def evaluate(self, all_strains):
        q = 3
        for i in range(self.n):
            strain = all_strains[q * i : q * i + q]
            sig_tr_i = self.mat.D @ (strain - self.eps_p[i])

            sig_cr, Ct, k, d_eps_p = self.mat.correct_stress(sig_tr_i, self.kappa[i], 1e-9, 20)
            # assignments:
            self.kappa1[i] = k # update history variable(s)
            self.stress[i] = sig_cr
            self.dstress[i] = Ct.flatten()
            self.deps_p[i] = d_eps_p # store change in the cumulated plastic strain

"""
######################################################
##             ABBAS PLASTICITY MODEL END
######################################################
"""

