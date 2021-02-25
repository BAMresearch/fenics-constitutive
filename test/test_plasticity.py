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


class NormVM:
    def __init__(self, constraint):
        self.ss_dim = c.q_dim(constraint)
        
        _fact = 6.0
        
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
        self.ss_dim = c.q_dim(constraint)
        
        if self.ss_dim == 1:
            self.D = np.array([self.E])
        else:
            self.mu, self.lamda = constitutive_coeffs(E=self.E, noo=self.noo, constraint=constraint)
            _fact = 1
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
        
class Yield_VM:
    def __init__(self, y0, constraint, H=0):
        self.y0 = y0 # yield stress
        self.constraint = constraint
        self.ss_dim = c.q_dim(constraint)
        self.H = H # isotropic hardening modulus
        self.vm_norm = NormVM(self.constraint)
        
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

"""
######################################################
##             ABBAS PLASTICITY MODEL END
######################################################
"""


class Plasticity:
    def __init__(self, mat):
        self.mat = mat
        self.q = mat.ss_dim

    def resize(self, n):  # just to match the interface
        self.n = n
        q = 3
        self.stress = np.zeros((self.n, q))
        self.dstress = np.zeros((self.n,  q * q))
        self.eps_p = np.zeros((self.n, q))
        self.kappa1 = np.zeros(self.n)
        self.kappa = np.zeros(self.n)
        self.deps_p = np.zeros((self.n, q))

    def update(self, all_strains):
        self.kappa = self.kappa1
        self.eps_p += self.deps_p

    def evaluate(self, all_strains):
        q = 3
        for i in range(self.n):
            strain = all_strains[q * i : q * i + q]
            sig_tr_i = self.mat.D @ (strain - self.eps_p[i])

            sig_cr, Ct, k, d_eps_p = self.mat.correct_stress(sig_tr=sig_tr_i, k0=self.kappa[i])
            # assignments:
            self.kappa1[i] = k # update history variable(s)
            self.stress[i] = sig_cr
            self.dstress[i] = Ct.flatten()
            self.deps_p[i] = d_eps_p # store change in the cumulated plastic strain

class TestPlasticity(unittest.TestCase):
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

        prm = c.Parameters(c.Constraint.PLANE_STRESS)
        prm.E = 1000.0
        prm.Et = prm.E / 1000.0
        prm.sig0 = 12.0
        # prm.H = 15.0 * prm.E * prm.Et / (prm.E - prm.Et)
        prm.nu = 0.3
        prm.deg_d = 3
        prm.deg_q = 4
        law = c.LinearElastic(prm.E, prm.nu, prm.constraint)

        mesh = df.RectangleMesh(
            df.Point(0, 0),
            df.Point(LX, LY),
            int(LX * mesh_resolution),
            int(LY * mesh_resolution),
        )

        yf = Yield_VM(prm.sig0, prm.constraint)
        mat = PlasticConsitutivePerfect(prm.E, prm.nu, prm.constraint, yf=yf)
        plasticity = Plasticity(mat)


        problem = c.MechanicsProblem(mesh, prm, law, base=plasticity)

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
    unittest.main()
