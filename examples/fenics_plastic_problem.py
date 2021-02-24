#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 2021

@author: ajafari

CHANGE made:
    Use:
        fenics_plastic_constitutive.eps_vector
    instead of:
        MechanicsProblem.eps()
"""

import dolfin as df
from . import helper as h
import numpy as np
from examples.fenics_plastic_constitutive import ss_dim, eps_vector

def spaces(mesh, deg_q, qdim):
    cell = mesh.ufl_cell()
    q = "Quadrature"
    QF = df.FiniteElement(q, cell, deg_q, quad_scheme="default")
    QV = df.VectorElement(q, cell, deg_q, quad_scheme="default", dim=qdim)
    QT = df.TensorElement(q, cell, deg_q, quad_scheme="default", shape=(qdim, qdim))
    return [df.FunctionSpace(mesh, Q) for Q in [QF, QV, QT]]

class MechanicsProblem(df.NonlinearProblem):
    def __init__(self, mesh, prm, constraint):
        df.NonlinearProblem.__init__(self)

        self.mesh = mesh
        self.prm = prm # discretization parameters

        # self.law = law

        # self.base = Base(self.law)

        # if mesh.geometric_dimension() != g_dim(prm.constraint):
        #     raise RuntimeError(
        #         f"The geometric dimension of the mesh does not match the {prm.constraint} constraint."
        #     )

        metadata = {"quadrature_degree": prm.deg_q, "quadrature_scheme": "default"}
        dxm = df.dx(metadata=metadata)

        # solution field
        self.V = df.VectorFunctionSpace(mesh, prm.element_family, degree=prm.deg_d)
        self.d = df.Function(self.V, name="displacement field")

        # generic quadrature function spaces
        self.constraint = constraint
        self.qdim = ss_dim(self.constraint)
        self.VQF, self.VQV, VQT = spaces(mesh, prm.deg_q, self.qdim)

        # quadrature function
        self.q_sigma = df.Function(self.VQV, name="current stresses")
        self.q_eps = df.Function(self.VQV, name="current strains")
        self.q_dsigma_deps = df.Function(VQT, name="stress-strain tangent")

        self.n_gauss_points = len(self.q_eps.vector().get_local()) // self.qdim
        # self.base.resize(self.n_gauss_points);

        dd, d_ = df.TrialFunction(self.V), df.TestFunction(self.V)

        # eps = self.eps
        # self.R = df.inner(eps(d_), self.q_sigma) * dxm
        # self.dR = df.inner(eps(dd), self.q_dsigma_deps * eps(d_)) * dxm        
        # self.calculate_eps = h.LocalProjector(eps(self.d, self.constraint), self.VQV, dxm)
        
        eps = eps_vector
        self.R = df.inner(eps(d_, self.constraint), self.q_sigma) * dxm
        self.dR = df.inner(eps(dd, self.constraint), self.q_dsigma_deps * eps(d_, self.constraint)) * dxm
        self.calculate_eps = h.LocalProjector(eps(self.d, self.constraint), self.VQV, dxm)

        self._assembler = None
        self._bcs = None

    def add_force_term(self, term):
        self.R -= term
        if self._bcs is not None:
            # update it to the new self.R!
            self.set_bcs(self._bcs)

    @property
    def Vd(self):
        """
        Return the function space for the displacements. This distinction is
        required for mixed problems where V != Vd.
        """
        return self.V

    @property
    def u(self):
        """
        Return the full (for all dofs) solution field. 
        """
        return self.d

    # def eps(self, u):
    #     e = df.sym(df.grad(u))
    #     dim = self.mesh.geometric_dimension()
    #     if dim == 1:
    #         return df.as_vector([e[0, 0]])

    #     if dim == 2:
    #         return df.as_vector([e[0, 0], e[1, 1], 2 * e[0, 1]])

    #     if dim == 3:
    #         return df.as_vector(
    #             [e[0, 0], e[1, 1], e[2, 2], 2 * e[1, 2], 2 * e[0, 2], 2 * e[0, 1]]
    #         )

    def evaluate_material(self):
        # project the strain and the nonlocal equivalent strains onto
        # their quadrature spaces and ...
        self.calculate_eps(self.q_eps)
        # self.base.evaluate(self.q_eps.vector().get_local())

        # ... and write the calculated values into their quadrature spaces.
        # h.set_q(self.q_sigma, self.base.stress)
        # h.set_q(self.q_dsigma_deps, self.base.dstress)

    def update(self):
        self.calculate_eps(self.q_eps)
        # self.base.update(self.q_eps.vector().get_local())


    def set_bcs(self, bcs):
        # Only now (with the bcs) can we initialize the _assembler
        # self._bcs = bcs
        self._assembler = df.SystemAssembler(self.dR, self.R, bcs)

    def F(self, b, x):
        if not self._assembler:
            raise RuntimeError("You need to `.set_bcs(bcs)` before the solve!")
        self.evaluate_material()
        self._assembler.assemble(b, x)

    def J(self, A, x):
        self._assembler.assemble(A)

    def solve(self, solver=None):
        if solver is None:
            solver = df.NewtonSolver()
        sol_output = solver.solve(self, self.u.vector())
        return sol_output, self.u

class PlasticProblem(MechanicsProblem):
    def __init__(self, mat, mesh, prm, constraint):
        self.mat = mat
        MechanicsProblem.__init__(self, mesh, prm, constraint)
        assert (self.qdim == self.mat.ss_dim)
        
        self.sig_num = np.zeros((self.n_gauss_points, self.qdim)) # this is just helpfull for assigning to self.q_sigma
        
        self.kappa1 = np.zeros(self.n_gauss_points) # Current history variable at last global NR iteration
        self.kappa = df.Function(self.VQF, name="Cumulated history variable")
        
        self.d_eps_p_num = np.zeros((self.n_gauss_points, self.qdim)) # Change of plastic strain at last global NR iteration
        self.eps_p = df.Function(self.VQV, name="Cumulated plastic strain")
        
        # Define and initiate tangent operator (elasto-plastic stiffness matrix)
        self.Ct_num = np.tile(self.mat.D.flatten(), self.n_gauss_points).reshape((self.n_gauss_points, self.qdim**2)) # initial value is the elastic stiffness at all Gauss-points
        h.set_q(self.q_dsigma_deps, self.Ct_num.flatten()) # assign the helper "Ct_num" to "q_dsigma_deps"
        
        self.time_varying_loadings = []
        
        df.parameters["form_compiler"]["representation"] = "quadrature"
        import warnings
        from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
        warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)
    
    def F(self, b, x):
        # project the corresponding strain to quadrature space
        self.calculate_eps(self.q_eps)
        # compute correct stress and Ct based on the updated strain "self.eps" (perform return-mapping, if needed)
        self.evaluate_material()
        # update the solver's residual
        self._assembler.assemble(b, x)

    def J(self, A, x):
        # update the solver's tangent operator
        self._assembler.assemble(A)

    def solve_t(self, t=0.0, max_iters=20, allow_nonconvergence_error=True, solver=None):
        for l in self.time_varying_loadings:
            l.t = t
        (_it, conv), _ = MechanicsProblem.solve(self, solver)
        if conv:
            print(f"    The time step t={t} converged after {_it} iteration(s).")
            self.update()
        else:
            print(f"    The time step t={t} did not converge.")
        return (_it, conv)
    
    def evaluate_material(self):
        """
        given:
            self.q_eps
        , we perform:
            the update of stress and Ct
        """
        eps = self.q_eps.vector().get_local().reshape((-1, self.qdim))
        eps_p = self.eps_p.vector().get_local().reshape((-1, self.qdim))
        
        # perform return-mapping (if needed) per individual Gauss-points
        for i in range(self.n_gauss_points):
            sig_tr_i = np.atleast_1d(self.mat.D @ (eps[i] - eps_p[i]))
            sig_cr, Ct, k, d_eps_p = self.mat.correct_stress(sig_tr=sig_tr_i, k0=self.kappa.vector()[i])
            # assignments:
            self.kappa1[i] = k # update history variable(s)
            self.sig_num[i] = sig_cr
            self.Ct_num[i] = Ct.flatten()
            self.d_eps_p_num[i] = d_eps_p # store change in the cumulated plastic strain
        
        # assign the helper "sig_num" to "q_sigma"
        h.set_q(self.q_sigma, self.sig_num.flatten())
        # assign the helper "Ct_num" to "Ct"
        h.set_q(self.q_dsigma_deps, self.Ct_num.flatten())
    
    def update(self): # over-write
        h.set_q(self.kappa, self.kappa1)
        h.set_q(self.eps_p, self.eps_p.vector()[:] + self.d_eps_p_num.flatten())
        
    def add_time_varying_loadings(self, e):
        self.time_varying_loadings.append(e)
        
    def get_F(self):
        return self.R
    
    def get_uu(self):
        return self.u