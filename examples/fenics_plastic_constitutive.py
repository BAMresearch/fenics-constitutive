#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 2021

@author: ajafari
"""

import dolfin as df
import numpy as np
from examples.helper import LocalProjector

## GLOBALLY DEFINED
_ss_vector = 'Voigt' # with (2) factor only in shear strains
# _ss_vector = 'Mandel' # with sqrt(2) factor in both shear strains and shear stresses

def ss_dim(constraint):
    constraint_switcher = {
                'UNIAXIAL': 1,
                'PLANE_STRESS': 3,
                'PLANE_STRAIN': 4,
                '3D': 6
             }
    _dim = constraint_switcher.get(constraint, "Invalid constraint given. Possible values are: " + str(constraint_switcher.keys()))
    return _dim

def eps_vector(v, constraint):
    _ss_dim = ss_dim(constraint)
    if _ss_dim==1:
        return df.grad(v)
    else:
        e = df.sym(df.grad(v))
        if _ss_vector == 'Voigt':
            _fact = 2
        elif _ss_vector == 'Mandel':
            _fact = 2 ** 0.5
        
        if _ss_dim==3: # for now, plane stress #################################
            return df.as_vector([e[0, 0], e[1, 1], _fact * e[0, 1]])
        elif _ss_dim==4: # for now, means plane strain, too. #######################
            return df.as_vector([e[0, 0], e[1, 1], 0, _fact * e[0, 1]])
        elif _ss_dim==6:
            return df.as_vector( [ e[0, 0], e[1, 1], e[2, 2] \
                                 , _fact * e[1, 2], _fact * e[0, 2], _fact * e[0, 1] ] )

def constitutive_coeffs(E=1000, noo=0.0, constraint='PLANE_STRESS'):
    lamda=(E*noo/(1+noo))/(1-2*noo)
    mu=E/(2*(1+noo))
    if constraint=='PLANE_STRESS':
        lamda = 2*mu*lamda/(lamda+2*mu)
    return mu, lamda

class ElasticConstitutive():
    def __init__(self, E, noo, constraint):
        self.E = E
        self.noo = noo
        self.constraint = constraint
        self.ss_dim = ss_dim(self.constraint)
        
        if self.ss_dim == 1:
            self.D = np.array([self.E])
        else:
            self.mu, self.lamda = constitutive_coeffs(E=self.E, noo=self.noo, constraint=constraint)
            if _ss_vector == 'Voigt':
                _fact = 1
            elif _ss_vector == 'Mandel':
                _fact = 2
            if self.ss_dim == 3:
                self.D = (self.E / (1 - self.noo ** 2)) * np.array([ [1, self.noo, 0], [self.noo, 1, 0], [0, 0, _fact * 0.5 * (1-self.noo) ] ])
            elif self.ss_dim == 4:
                self.D = np.array([
                        [2 * self.mu + self.lamda, self.lamda, self.lamda, 0],
                        [self.lamda, 2 * self.mu + self.lamda, self.lamda, 0],
                        [self.lamda, self.lamda, 2 * self.mu + self.lamda, 0],
                        [0, 0, 0, _fact * self.mu],
                    ])
            elif self.ss_dim == 6:
                self.D = np.array([
                        [2 * self.mu + self.lamda, self.lamda, self.lamda, 0, 0, 0],
                        [self.lamda, 2 * self.mu + self.lamda, self.lamda, 0, 0, 0],
                        [self.lamda, self.lamda, 2 * self.mu + self.lamda, 0, 0, 0],
                        [0, 0, 0, _fact * self.mu, 0, 0],
                        [0, 0, 0, 0, _fact * self.mu, 0],
                        [0, 0, 0, 0, 0, _fact * self.mu],
                    ])

class NormVM:
    def __init__(self, constraint, stress_norm):
        self.constraint = constraint
        self.ss_dim = ss_dim(self.constraint)
        self.stress_norm = stress_norm # True: norm of stress, False: norm of strain
        
        if _ss_vector == 'Voigt':
            if self.stress_norm:
                _fact = 6.0
            else: # for strain
                _fact = 1.5
        elif _ss_vector == 'Mandel':
            _fact = 3.0
        
        if self.ss_dim == 1:
            self.P = np.array([2/3])
        elif self.ss_dim==3:
            self.P = (1/3) * np.array([[2, -1, 0], [-1, 2, 0], [0, 0, _fact]])
        elif self.ss_dim==4:
            self.P = (1/3) * np.array([[2, -1, -1, 0], [-1, 2, -1, 0], [-1, -1, 2, 0], [0, 0, 0, _fact]])
        elif self.ss_dim==6:
            self.P = (1/3) * np.array( [ [2, -1, -1, 0, 0, 0], [-1, 2, -1, 0, 0, 0], [-1, -1, 2, 0, 0, 0] \
                                       , [0, 0, 0, _fact, 0, 0], [0, 0, 0, 0, _fact, 0], [0, 0, 0, 0, 0, _fact] ] )
    def __call__(self, ss):
        """
        ss: stress/strain vector for which
        se: the VM norm
        and
        m: the derivative of se w.r.t. ss
        are computed/returned.
        """
        if self.ss_dim==1:
            se = abs(ss)
            m = 1 # default for the case _norm=0
            if se>0:
                m = np.sign(ss[0])
        else:
            assert (len(ss)==self.ss_dim)
            se = (1.5 * ss.T @ self.P @ ss) ** 0.5
            if se == 0:
                m = np.zeros(self.ss_dim)  # using central difference method
                # m = np.sqrt(1.5 * np.diag(self.P)) # using forward difference method
            else:
                m = (1.5 / se) * self.P @ ss
        return se, m

class RateIndependentHistory:
    def __init__(self, p=None, dp_dsig=None, dp_dk=None):
        """
        The goal of this class is to hardcode the consistent provide of enough information about a rate-independent history evolution.
        This includes a callable (p_and diffs) with:
            INPUTS:
                sigma: effective (elastic) stress
                kappa: internal variable(s)
            OUTPUTS:
                p: (depending on material) the intended rate-independent function "p(sigma, kappa)"
                    , such that: kappa_dot = lamda_dot * p , where lamda is the well-known "plastic multiplier".
                    ---> this is a "rate-independent" evolution of "kappa".
                dp_dsig: accordingly the derivative of "p" w.r.t. sigma (with same inputs)
                dp_dk: accordingly the derivative of "p" w.r.t. kappa (with same inputs)
        IMPORTANT:
            All outputs must be in form of 2-D np.array with a full shape (m,n), e.g. scalar values are of shape (1,1)
        """
        if p is None: # default is p=1, i.e. kappa_dot = lamda_dot
            def p_and_diffs(sigma, kappa):
                return np.array([[1.0]]), np.array([[0.0]]), np.array([[0.0]])
        else:
            assert(dp_dsig is not None)
            assert(dp_dk is not None)
            def p_and_diffs(sigma, kappa):
                return p(sigma, kappa), dp_dsig(sigma, kappa), dp_dk(sigma, kappa)
        self.p_and_diffs = p_and_diffs
    def __call__(self, sigma, kappa):
        """
        sigma: effective stress
        kappa: internal variable(s)
        returns:
            p, dp_dsig, dp_dk (all explained in constructor)
        """
        return self.p_and_diffs(sigma, kappa)

class RateIndependentHistory_PlasticWork(RateIndependentHistory):
    def __init__(self, yield_function):
        self.yf = yield_function
        def p_and_diffs(sigma, kappa):
            _, m, dm, _, mk = self.yf(sigma, kappa)
            p = np.array([[np.dot(sigma, m)]])
            dp_dsig = (m + dm.T @ sigma).reshape((1, -1))
            dp_dk = np.array([[np.dot(sigma, mk)]])
            return p, dp_dsig, dp_dk
        self.p_and_diffs = p_and_diffs
    def __call__(self, sigma, kappa):
        return self.p_and_diffs(sigma, kappa)

class PlasticConsitutivePerfect(ElasticConstitutive):
    def __init__(self, E, nu, constraint, yf):
        """
        yf: a callable with:
            argument:      stress state
            return values: f: yield surface evaluation
                           m: flow vector; e.g. with associated flow rule: the first derivative of yield surface w.r.t. stress
                           dm: derivative of flow vector w.r.t. stress; e.g. with associated flow rule: second derivative of yield surface w.r.t. stress
        """
        super().__init__(E=E, noo=nu, constraint=constraint)
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
        assert(isinstance(ri, RateIndependentHistory))
        self.ri = ri # ri: rate-independent
    
    def correct_stress(self, sig_tr, k0=0.0, _Ct=True, tol=1e-9, max_iters=20):
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
                
class Yield_VM:
    def __init__(self, y0, constraint, H=0):
        self.y0 = y0 # yield stress
        self.constraint = constraint
        self.ss_dim = ss_dim(self.constraint)
        self.H = H # isotropic hardening modulus
        self.vm_norm = NormVM(self.constraint, stress_norm=True)
        
    def __call__(self, stress, kappa=0):
        """
        Evaluate the yield function quantities at a specific stress level (as a vector):
            f: yield function itself
            m: flow vector; derivative of "f" w.r.t. stress (associated flow rule)
            dm: derivative of flow vector w.r.t. stress; second derivative of "f" w.r.t. stress
            fk: derivative of "f" w.r.t. kappa
            mk: derivative of "m" w.r.t. kappa
        The given stress vector must be consistent with self.ss_dim
        kappa: history variable(s), here related to isotropic hardening
        """
        assert (len(stress) == self.ss_dim)
        se, m = self.vm_norm(stress)
        f = se - (self.y0 + self.H * kappa)
        fk = - self.H
        if self.ss_dim==1:
            dm = 0.0 # no dependency on "self.H"
            mk = 0.0
        else:
            if se ==0:
                dm = None # no needed in such a case
            else:
                dm = (6 * se * self.vm_norm.P - 6 * np.outer(self.vm_norm.P @ stress, m)) / (4 * se ** 2) # no dependency on "self.H"
            mk = np.array(len(stress) * [0.0])
        return f, np.atleast_1d(m), np.atleast_2d(dm), fk, np.atleast_1d(mk)
