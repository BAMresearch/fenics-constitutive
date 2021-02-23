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

def epsilon(u, _dim=None):
    if _dim==None:
        _dim = u.geometric_dimension()
    if _dim==1:
        return df.grad(u)
    else:
        return df.sym(df.grad(u))

def constitutive_coeffs(E=1000, noo=0.0, dim=2, stress_free=False):
    lamda=(E*noo/(1+noo))/(1-2*noo)
    mu=E/(2*(1+noo))
    if dim==2 and stress_free:
        lamda = 2*mu*lamda/(lamda+2*mu)
    return mu, lamda

class ElasticConstitutive():
    def __init__(self, dim, E, noo, stress_free=False):
        self.dim = dim
        self.E = E
        self.noo = noo
        self.stress_free = stress_free
        
        if self.dim == 1:
            self.ss_dim = 1
            self.D = np.array([self.E])
        else:
            self.mu, self.lamda = constitutive_coeffs(E=self.E, noo=self.noo, dim=self.dim, stress_free=self.stress_free)
            if _ss_vector == 'Voigt':
                _fact = 1
            elif _ss_vector == 'Mandel':
                _fact = 2
            if self.dim == 2:
                if self.stress_free: # plane stress
                    self.ss_dim = 3
                    self.D = (self.E / (1 - self.noo ** 2)) * np.array([ [1, self.noo, 0], [self.noo, 1, 0], [0, 0, _fact * 0.5 * (1-self.noo) ] ])
                else: # plane strain
                    self.ss_dim = 4
                    self.D = np.array([
                            [2 * self.mu + self.lamda, self.lamda, self.lamda, 0],
                            [self.lamda, 2 * self.mu + self.lamda, self.lamda, 0],
                            [self.lamda, self.lamda, 2 * self.mu + self.lamda, 0],
                            [0, 0, 0, _fact * self.mu],
                        ])
            elif self.dim == 3:
                self.ss_dim = 6
                self.D = np.array([
                        [2 * self.mu + self.lamda, self.lamda, self.lamda, 0, 0, 0],
                        [self.lamda, 2 * self.mu + self.lamda, self.lamda, 0, 0, 0],
                        [self.lamda, self.lamda, 2 * self.mu + self.lamda, 0, 0, 0],
                        [0, 0, 0, _fact * self.mu, 0, 0],
                        [0, 0, 0, 0, _fact * self.mu, 0],
                        [0, 0, 0, 0, 0, _fact * self.mu],
                    ])

    def sigma(self, u, K=None):
        eps_u = epsilon(u, self.dim)
        if self.dim == 1:
            return self.E * eps_u
        else:
            return self.lamda * df.tr(eps_u) * df.Identity(self.dim) + 2 * self.mu * eps_u

def eps_vector(v, ss_dim):
    if ss_dim==1:
        return df.grad(v)
        # return df.as_vector([df.grad(v)[0, 0]])
    else:
        e = df.sym(df.grad(v))
        if _ss_vector == 'Voigt':
            _fact = 2
        elif _ss_vector == 'Mandel':
            _fact = 2 ** 0.5
        
        if ss_dim==3: # for now, plane stress #################################
            return df.as_vector([e[0, 0], e[1, 1], _fact * e[0, 1]])
        elif ss_dim==4: # for now, means plane strain, too. #######################
            return df.as_vector([e[0, 0], e[1, 1], 0, _fact * e[0, 1]])
        elif ss_dim==6:
            return df.as_vector( [ e[0, 0], e[1, 1], e[2, 2] \
                                 , _fact * e[1, 2], _fact * e[0, 2], _fact * e[0, 1] ] )
class NormVM:
    def __init__(self, dim, stress_norm, stress_free=True):
        self.dim = dim
        self.stress_norm = stress_norm # True: norm of stress, False: norm of strain
        self.stress_free = stress_free # by default, plane-stress
        
        if _ss_vector == 'Voigt':
            if self.stress_norm:
                _fact = 6.0
            else: # for strain
                _fact = 1.5
        elif _ss_vector == 'Mandel':
            _fact = 3.0
        
        if self.dim == 1:
            self.P = np.array([2/3])
        elif self.dim==2:
            if self.stress_free:
                self.P = (1/3) * np.array([[2, -1, 0], [-1, 2, 0], [0, 0, _fact]])
            else: # plane strain
                pass # self.P = ...  to be set
        elif self.dim==3:
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
        if self.dim==1:
            se = abs(ss)
            m = 1 # default for the case _norm=0
            if se>0:
                m = np.sign(ss[0])
        else:
            if self.dim==2:
                if self.stress_free:
                    assert len(ss)==3
                else: # plane strain
                    assert len(ss)==4 ## ???
            elif self.dim==3:
                assert len(ss)==6
            se = (1.5 * ss.T @ self.P @ ss) ** 0.5
            if se == 0:
                m = np.zeros(len(ss))  # using central difference method
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
    def __init__(self, E, dim, nu, yf, stress_free=True):
        """
        yf: a callable with:
            argument:      stress state
            return values: f: yield surface evaluation
                           m: flow vector; e.g. with associated flow rule: the first derivative of yield surface w.r.t. stress
                           dm: derivative of flow vector w.r.t. stress; e.g. with associated flow rule: second derivative of yield surface w.r.t. stress
        """
        super().__init__(E=E, dim=dim, noo=nu, stress_free=stress_free)
        self.yf = yf
        if hasattr(self.yf, 'dim'):
            assert(self.dim == self.yf.dim)
    
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
    def __init__(self, E, dim, nu, yf, ri, stress_free=True):
        """
        ri: an instance of RateIndependentHistory representing evolution of history variables
        , which is based on:
            kappa_dot = lamda_dot * p(sigma, kappa), where "p" is a plastic modulus function
        """
        super().__init__(E, dim, nu, yf, stress_free)
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
    def __init__(self, y0, dim, H=0, stress_free=True):
        self.y0 = y0 # yield stress
        self.dim = dim
        self.H = H # isotropic hardening modulus
        self.vm_norm = NormVM(dim, stress_norm=True, stress_free=stress_free)
        
    def __call__(self, stress, kappa=0):
        """
        Evaluate the yield function quantities at a specific stress level (as a vector):
            f: yield function itself
            m: flow vector; derivative of "f" w.r.t. stress (associated flow rule)
            dm: derivative of flow vector w.r.t. stress; second derivative of "f" w.r.t. stress
            fk: derivative of "f" w.r.t. kappa
            mk: derivative of "m" w.r.t. kappa
        The given stress vector must be consistent with self.dim and self.stress_free
        kappa: history variable(s), here related to isotropic hardening
        """
        se, m = self.vm_norm(stress)
        f = se - (self.y0 + self.H * kappa)
        fk = - self.H
        if self.dim==1:
            dm = 0.0 # no dependency on "self.H"
            mk = 0.0
        else:
            if se ==0:
                dm = None # no needed in such a case
            else:
                dm = (6 * se * self.vm_norm.P - 6 * np.outer(self.vm_norm.P @ stress, m)) / (4 * se ** 2) # no dependency on "self.H"
            mk = np.array(len(stress) * [0.0])
        return f, np.atleast_1d(m), np.atleast_2d(dm), fk, np.atleast_1d(mk)

class FenicsProblem:
    def __init__(self, mat, mesh, fen_config, dep_dim=None):
        self.mat = mat
        self.mesh = mesh
        self.el_family = fen_config.el_family
        self.shF_degree_u = fen_config.shF_degree_u
        
        self.geo_dim = self.mesh.geometry().dim() # Geometrical dimension of mesh
        if dep_dim is None:
            self.dep_dim = self.geo_dim # The dimension of dependent variables
        
        self.bcs_DR = [] # list of Dirichlet BCs
        self.bcs_DR_measures = [] # list of measures
        self.bcs_DR_dofs = [] # list of DOFs of Dirichlet BCs
        
        self.bcs_DR_inhom = [] # list of inhomogeneous Dirichlet BCs
        self.bcs_DR_inhom_measures = [] # list of measures
        self.bcs_DR_inhom_dofs = [] # list of DOFs of inhomogeneous Dirichlet BCs
        
    def build_variational_functionals(self, f=None, integ_degree=None):
        if f is None:
            if self.dep_dim == 1:
                f = df.Constant(0.0)
            else:
                f = df.Constant(self.dep_dim * (0.0, ))
        if integ_degree is None:
            self.dxm = df.dx(self.mesh)
            self.integ_degree = 'default'
        else:
            assert(type(integ_degree) == int)
            self.dxm = self.quad_measure(integration_degree = integ_degree)
            self.integ_degree = integ_degree
            df.parameters["form_compiler"]["quadrature_degree"] = self.integ_degree
        self.discretize()
        return f
    
    def quad_measure(self, integration_degree=1, domain=None):
        if domain is None:
            domain = self.mesh
        md = {'quadrature_degree': integration_degree, 'quadrature_scheme': 'default'}
        return df.dx(domain = domain, metadata = md)
    
    def discretize(self):
        pass
    
    def build_solver(self, time_varying_loadings=[]):
        self.time_varying_loadings = time_varying_loadings
    
    def solve(self):
        pass
        
    def _todo_after_convergence(self):
        pass
    
    def get_F_and_u(self):
        pass
    
    def get_uu(self, _deepcopy=True):
        pass
    
    def get_iu(self, _collapse=False):
        pass
    
    def reset_fields(self):
        pass

class FenicsPlastic(FenicsProblem, df.NonlinearProblem):
    def __init__(self, mat, mesh, fen_config, dep_dim=None):
        FenicsProblem.__init__(self, mat, mesh, fen_config, dep_dim)
        df.NonlinearProblem.__init__(self)
    
    def build_variational_functionals(self, f=None, integ_degree=None):
        self.hist_storage = 'quadrature' # always is quadrature
        if integ_degree is None:
            integ_degree = self.shF_degree_u + 1
        f = FenicsProblem.build_variational_functionals(self, f, integ_degree) # includes a call for discretize method
        
        df.parameters["form_compiler"]["representation"] = "quadrature"
        import warnings
        from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
        warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)
        
        self.a_Newton = df.inner(eps_vector(self.v, self.mat.ss_dim), df.dot(self.Ct, eps_vector(self.u_, self.mat.ss_dim))) * self.dxm
        self.res = ( df.inner(eps_vector(self.u_, self.mat.ss_dim), self.sig) - df.inner(f, self.u_) ) * self.dxm
        
    def discretize(self):
        ### Nodal spaces / functions
        if self.dep_dim == 1:
            elem_u = df.FiniteElement(self.el_family, self.mesh.ufl_cell(), self.shF_degree_u)
        else:
            elem_u = df.VectorElement(self.el_family, self.mesh.ufl_cell(), self.shF_degree_u, dim=self.dep_dim)
        self.i_u = df.FunctionSpace(self.mesh, elem_u)
        # Define functions
        self.u = df.Function(self.i_u, name="Displacement at last global NR iteration")
        self.v = df.TrialFunction(self.i_u)
        self.u_ = df.TestFunction(self.i_u)
        
        ### Quadrature spaces / functions
        # for sigma and strain
        elem_ss = df.VectorElement("Quadrature", self.mesh.ufl_cell(), degree=self.integ_degree, dim=self.mat.ss_dim, quad_scheme="default")
        self.i_ss = df.FunctionSpace(self.mesh, elem_ss)
        # for scalar history variables on gauss points (for now: scalar)
        elem_scalar_gauss = df.FiniteElement("Quadrature", self.mesh.ufl_cell(), degree=self.integ_degree, quad_scheme="default")
        self.i_hist = df.FunctionSpace(self.mesh, elem_scalar_gauss)
        # for tangent matrix
        elem_tensor = df.TensorElement("Quadrature", self.mesh.ufl_cell(), degree=self.integ_degree, shape=(self.mat.ss_dim, self.mat.ss_dim), quad_scheme="default")
        i_tensor = df.FunctionSpace(self.mesh, elem_tensor)
        
        self.ngauss = self.i_hist.dim() # get total number of gauss points
        
        # Define functions based on Quadrature spaces
        self.sig = df.Function(self.i_ss, name="Stress")
        self.eps = df.Function(self.i_ss, name="Strain")
        self.sig_num = np.zeros((self.ngauss, self.mat.ss_dim)) # this is just helpfull for assigning to self.sig
        
        self.kappa1 = np.zeros(self.ngauss) # Current history variable at last global NR iteration
        self.kappa = df.Function(self.i_hist, name="Cumulated history variable")
        
        self.d_eps_p_num = np.zeros((self.ngauss, self.mat.ss_dim)) # Change of plastic strain at last global NR iteration
        self.eps_p = df.Function(self.i_ss, name="Cumulated plastic strain")
        
        # Define and initiate tangent operator (elasto-plastic stiffness matrix)
        self.Ct = df.Function(i_tensor, name="Tangent operator")
        self.Ct_num = np.tile(self.mat.D.flatten(), self.ngauss).reshape((self.ngauss, self.mat.ss_dim**2)) # initial value is the elastic stiffness at all Gauss-points
        self.Ct.vector().set_local(self.Ct_num.flatten()) # assign the helper "Ct_num" to "Ct"
    
    def build_solver(self, time_varying_loadings=[], tol=1e-12, solver=None):
        super().build_solver(time_varying_loadings)
        self.projector_eps = LocalProjector(eps_vector(self.u, self.mat.ss_dim), self.i_ss, self.dxm)
        if len(self.bcs_DR + self.bcs_DR_inhom) == 0:
            print('WARNING: No boundary conditions have been set to the FEniCS problem.')
        self.assembler = df.SystemAssembler(self.a_Newton, self.res, self.bcs_DR + self.bcs_DR_inhom)
        if solver is None:
            solver = df.NewtonSolver()
        self.solver = solver
    
    def F(self, b, x):
        # project the corresponding strain to quadrature space
        self.projector_eps(self.eps)
        # compute correct stress and Ct based on the updated strain "self.eps" (perform return-mapping, if needed)
        self._correct_stress_and_Ct()
        # update the solver's residual
        self.assembler.assemble(b, x)

    def J(self, A, x):
        # update the solver's tangent operator
        self.assembler.assemble(A)

    def solve(self, t=0.0, max_iters=20, allow_nonconvergence_error=True):
        if not hasattr(self, 'solver'):
            raise RuntimeError("Please call FenicsPlastic.build_solver() method before solving the problem.")
        for l in self.time_varying_loadings:
            l.t = t
        _it, conv = self.solver.solve(self, self.u.vector())
        if conv:
            print(f"    The time step t={t} converged after {_it} iteration(s).")
            self._todo_after_convergence()
        else:
            print(f"    The time step t={t} did not converge.")
        return (_it, conv)
    
    def _correct_stress_and_Ct(self):
        """
        given:
            self.eps
        , we perform:
            the update of stress and Ct
        """
        eps = self.eps.vector().get_local().reshape((-1, self.mat.ss_dim))
        eps_p = self.eps_p.vector().get_local().reshape((-1, self.mat.ss_dim))
        
        # perform return-mapping (if needed) per individual Gauss-points
        for i in range(self.ngauss):
            sig_tr_i = np.atleast_1d(self.mat.D @ (eps[i] - eps_p[i]))
            sig_cr, Ct, k, d_eps_p = self.mat.correct_stress(sig_tr=sig_tr_i, k0=self.kappa.vector()[i])
            # assignments:
            self.kappa1[i] = k # update history variable(s)
            self.sig_num[i] = sig_cr
            self.Ct_num[i] = Ct.flatten()
            self.d_eps_p_num[i] = d_eps_p # store change in the cumulated plastic strain
        
        # assign the helper "sig_num" to "sig"
        self.sig.vector().set_local(self.sig_num.flatten())
        # assign the helper "Ct_num" to "Ct"
        self.Ct.vector().set_local(self.Ct_num.flatten())
    
    def _todo_after_convergence(self):
        self.kappa.vector()[:] = self.kappa1
        self.eps_p.vector()[:] = self.eps_p.vector()[:] + self.d_eps_p_num.flatten()
        
    def get_F_and_u(self):
        return self.res, self.u
    
    def get_uu(self):
        return self.u
