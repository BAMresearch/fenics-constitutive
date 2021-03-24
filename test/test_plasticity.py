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
##             ABBAS PLASTICITY MODEL BEGIN
######################################################
"""

class RateIndependentHistory_PlasticWork(c.RateIndependentHistory):
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

class PlasticConsitutivePerfect(c.ElasticConstitutive):
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

def revise_prm(prm):
    prm.E = 1000.0
    prm.Et = prm.E / 100.0
    prm.sig0 = 12.0
    prm.H = 15.0 * prm.E * prm.Et / (prm.E - prm.Et)
    prm.nu = 0.3
    prm.deg_d = 3
    prm.deg_q = 4
    return prm

def plasticity_law(prm):
    prm = revise_prm(prm)
    yf = c.YieldVM(prm.sig0, prm.constraint, prm.H)
    ri = c.RateIndependentHistory()
    ## law (at GP-level) in PYTHON
    # law = PlasticConsitutiveRateIndependentHistory(prm.E, prm.nu, prm.constraint, yf=yf, ri=ri)
    ## law (at GP-level) in C++
    law = c.PlasticConsitutiveRateIndependentHistory(prm.E, prm.nu, prm.constraint, yf, ri)
    return law

class TestPlasticity(unittest.TestCase):
    
    def test_at_one_point(self):
        """
        Given the adjusted parameters:
            
            PLANE_STRESS
            prm.E = 1000.0
            prm.Et = prm.E / 100.0
            prm.sig0 = 12.0
            prm.H = 15.0 * prm.E * prm.Et / (prm.E - prm.Et)
            prm.nu = 0.3
        
        , the following inputs:
            
            strain = array([ 0.01698246, -0.00421753,  0.00357475])
            k0 = 0.0 (default value)
                (this also means: sig_tr = array([17.27165151,  0.96396865,  1.37490339]) )
        
        , must result in:
        
            sig_cr = array([13.206089  ,  1.47886298,  0.98872433])
            Ct = array([[280.44279754, 338.97481295, -51.57281077],
               [338.97481295, 848.94226837,   3.64252814],
               [-51.57281077,   3.64252814, 271.93046544]])
            k = array([0.00428167])
            d_eps_p = array([ 0.00422003, -0.00173456,  0.00100407])
        """ 
        prm = c.Parameters(c.Constraint.PLANE_STRESS)
        law = law_from_prm(prm)
        law.resize(1)
        strain = np.array([ 0.01698246, -0.00421753,  0.00357475])
        sig_cr = np.array([13.206089  ,  1.47886298,  0.98872433])
        Ct = np.array([[280.44279754, 338.97481295, -51.57281077],
               [338.97481295, 848.94226837,   3.64252814],
               [-51.57281077,   3.64252814, 271.93046544]])
        gp_id = 0; k0 = 0.0;
        sigma, dsigma = law.evaluate(strain, gp_id)
        sigma_c, dsigma_c, _, _ = law.correct_stress(law.D@strain, k0, 1e-9, 20)
        self.assertLess( np.linalg.norm(sigma-sigma_c), 1e-9)
        self.assertLess( np.linalg.norm(dsigma-dsigma_c), 1e-9)
        self.assertLess( np.linalg.norm(sigma-sig_cr), 1e-5)
        self.assertLess( np.linalg.norm(dsigma-Ct), 5e-4)
    
    def test_bending(self):
        # return
        LX = 6.0
        LY = 0.5
        LZ = 0.5
        mesh_resolution = 6.0

        def loading(t):
            level = 4 * LZ
            N = 1.5
            return level * np.sin(N * t * 2 * np.pi)
        # show_loading(loading)  # if you really insist on it :P
        
        mesh = df.RectangleMesh(
            df.Point(0, 0),
            df.Point(LX, LY),
            int(LX * mesh_resolution),
            int(LY * mesh_resolution),
        )
        
        prm = c.Parameters(c.Constraint.PLANE_STRESS)
        law = plasticity_law(prm)
        
        ## ip-loop in PYTHON
        # problem = c.MechanicsProblem(mesh, prm, law=None, iploop=PlasticityIPLoopInPython(law))
        ## ip-loop in C++
        problem = c.MechanicsProblem(mesh, prm, law=law)

        left = boundary.plane_at(0.0)
        right_top = boundary.point_at((LX, LY))
        bc_expr = df.Expression("u", degree=0, u=0)

        bcs = []
        bcs.append(
            df.DirichletBC(problem.Vd.sub(1), bc_expr, right_top, method="pointwise")
        )
        bcs.append(df.DirichletBC(problem.Vd, (0, 0), left))
        problem.set_bcs(bcs)

        linear_solver = df.LUSolver("mumps")
        solver = df.NewtonSolver(
            df.MPI.comm_world, linear_solver, df.PETScFactory.instance()
        )
        solver.parameters["linear_solver"] = "mumps"
        solver.parameters["maximum_iterations"] = 10
        solver.parameters["error_on_nonconvergence"] = False

        def solve(t, dt):
            print(t, dt)
            bc_expr.u = loading(t)
            return solver.solve(problem, problem.u.vector())

        ld = c.helper.LoadDisplacementCurve(bcs[0])
        ld.show()

        if not ld.is_root:
            set_log_level(LogLevel.ERROR)

        fff = df.XDMFFile("output.xdmf")
        fff.parameters["functions_share_mesh"] = True
        fff.parameters["flush_output"] = True

        def pp(t):
            problem.update()

            # this fixes XDMF time stamps
            import locale

            locale.setlocale(locale.LC_NUMERIC, "en_US.UTF-8")
            fff.write(problem.u, t)

            ld(t, df.assemble(problem.R))

        TimeStepper(solve, pp, problem.u).dt_max(0.02).adaptive(1.0, dt=0.01)

if __name__ == "__main__":
    ### ALL TESTs
    unittest.main()
    
    ### SELECTIVE
    # tests = TestPlasticity()
    # tests.test_bending()
    # tests.test_at_one_point()
