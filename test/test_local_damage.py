import unittest
import dolfin as df
import numpy as np
from fenics_helpers import boundary
from fenics_helpers.timestepping import TimeStepper
import constitutive as c

def cdf2(f, x, delta):
    f0 = f(x)
    N = len(x)
    f_cdf = np.empty((N,N))
    for i in range(N):
        d = np.zeros_like(x)
        d[i] = delta
        f_cdf[:,i] = (f(x + d) - f(x - d)) / (2 * delta)
    return f_cdf


class TestUniaxial(unittest.TestCase):
    def test_tangent(self):
        prm = c.Parameters(c.Constraint.PLANE_STRESS)
        law = c.LocalDamage(prm.E, prm.nu, prm.constraint, prm.ft, prm.alpha, prm.gf, prm.k)
        law.resize(1)

        only_sigma = lambda x : law.evaluate(x, 0)[0]
        
        np.random.seed(6174)
        for i in range(42):
            strain = np.random.random(3)
            sigma, dsigma = law.evaluate(strain, 0)
            dsigma_cdf = cdf2(only_sigma, strain, 1.e-6)

            self.assertLess(np.linalg.norm(dsigma - dsigma_cdf), 1.e-4)

    def test_zero(self):
        prm = c.Parameters(c.Constraint.PLANE_STRESS)
        law = c.LocalDamage(prm.E, prm.nu, prm.constraint, prm.ft, prm.alpha, prm.gf, prm.k)
        law.resize(1)
        sigma, dsigma = law.evaluate([0,0,0])
        self.assertLess(np.max(np.abs(sigma)), 1.e-10)
        self.assertFalse(np.any(np.isnan(dsigma)))

    def test_bending(self):
        # return
        LX = 200
        LY = 30
        LX_load = 50

        prm = c.Parameters(c.Constraint.PLANE_STRAIN)
        prm.gf = 0.2
        prm.deg_d = 1
        law = c.LocalDamage(prm.E, prm.nu, prm.constraint, prm.ft, prm.alpha, prm.gf, prm.k)
        # law = c.LinearElastic(prm.E, prm.nu, prm.constraint)

        mesh = df.RectangleMesh(df.Point(0, 0), df.Point(LX, LY), 100, 15, "crossed")
        problem = c.MechanicsProblem(mesh, prm, law)

        left = boundary.point_at((0.0, 0.0), eps=0.1)
        right = boundary.point_at((LX, 0.0), eps=0.1)
        top = boundary.within_range([(LX - LX_load) / 2.0, LY], [(LX + LX_load) / 2, LY], eps=0.1)
        bc_expr = df.Expression("d*t", degree=0, t=0, d=-3)
        bcs = []
        bcs.append(df.DirichletBC(problem.Vd.sub(1), bc_expr, top))
        bcs.append(df.DirichletBC(problem.Vd.sub(0), 0.0, left, method="pointwise"))
        bcs.append(df.DirichletBC(problem.Vd.sub(1), 0.0, left, method="pointwise"))
        bcs.append(df.DirichletBC(problem.Vd.sub(1), 0.0, right, method="pointwise"))
        problem.set_bcs(bcs)
       
        linear_solver=df.LUSolver("mumps") 
        solver = df.NewtonSolver(df.MPI.comm_world, linear_solver, df.PETScFactory.instance())
        solver.parameters["linear_solver"] = "mumps"
        solver.parameters["maximum_iterations"] = 10
        solver.parameters["error_on_nonconvergence"] = False

        def solve(t, dt):
            print(t, dt)
            bc_expr.t = t
            try:
                return solver.solve(problem, problem.u.vector())
            except:
                return -1, False

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

        TimeStepper(solve, pp, problem.u).adaptive(1.0, dt=0.01)

    def test_uniaxial(self):
        return
        prm = c.Parameters(c.Constraint.UNIAXIAL_STRESS)
        prm.deg_d = 1
        prm.alpha = 0.999
        prm.gf = 0.01
        k0 = 2.e-4
        law = c.LocalDamage(prm.E, prm.nu, prm.constraint, prm.E * k0, prm.alpha, prm.gf, prm.k)

        mesh = df.UnitIntervalMesh(1)
        problem = c.MechanicsProblem(mesh, prm, law)

        bc0 = df.DirichletBC(problem.Vd, [0], boundary.plane_at(0))
        bc_expr = df.Expression(["u"], u=0, degree=0)
        bc1 = df.DirichletBC(problem.Vd, bc_expr, boundary.plane_at(1))

        problem.set_bcs([bc0, bc1])

        ld = c.helper.LoadDisplacementCurve(bc1)

        # ld.show()

        solver = df.NewtonSolver()
        solver.parameters["linear_solver"] = "mumps"
        solver.parameters["maximum_iterations"] = 10
        solver.parameters["error_on_nonconvergence"] = False


        u_max = 100 * k0
        for u in np.linspace(0, u_max, 101):
            bc_expr.u = u
            converged = solver.solve(problem, problem.u.vector())
            assert converged
            problem.update()
            ld(u, df.assemble(problem.R))

        GF = np.trapz(ld.load, ld.disp)
        self.assertAlmostEqual(GF, 0.5 * k0**2 * prm.E + prm.gf, delta=prm.gf/100) 

if __name__ == "__main__":
    unittest.main()
