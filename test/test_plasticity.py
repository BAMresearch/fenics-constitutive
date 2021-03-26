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
    yf = c.YieldVM(prm.sig0, prm.constraint, prm.H)
    ri = c.RateIndependentHistory()
    ## law (at GP-level) in PYTHON
    # law = PlasticConsitutiveRateIndependentHistory(prm.E, prm.nu, prm.constraint, yf=yf, ri=ri)
    ## law (at GP-level) in C++
    law = c.PlasticConsitutiveRateIndependentHistory(
        prm.E, prm.nu, prm.constraint, yf, ri
    )

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
        law = plasticity_law(revise_prm(prm))
        law.resize(1)
        strain = np.array([0.01698246, -0.00421753, 0.00357475])
        sig_cr = np.array([13.206089, 1.47886298, 0.98872433])
        Ct = np.array(
            [
                [280.44279754, 338.97481295, -51.57281077],
                [338.97481295, 848.94226837, 3.64252814],
                [-51.57281077, 3.64252814, 271.93046544],
            ]
        )
        gp_id = 0
        k0 = 0.0
        sigma, dsigma = law.evaluate(strain, gp_id)
        sigma_c, dsigma_c, _, _ = law.correct_stress(law.D @ strain, k0, 1e-9, 20)
        self.assertLess(np.linalg.norm(sigma - sigma_c), 1e-9)
        self.assertLess(np.linalg.norm(dsigma - dsigma_c), 1e-9)
        self.assertLess(np.linalg.norm(sigma - sig_cr), 1e-5)
        self.assertLess(np.linalg.norm(dsigma - Ct), 5e-4)

    def test_1d(self):
        mesh = df.UnitIntervalMesh(5)
        prm = c.Parameters(c.Constraint.UNIAXIAL_STRESS)
        # I think that this test should fail for UNIAXIAL_STRAIN...
        prm.E = 17.0
        prm.Et = 12.0
        prm.sig0 = 42.0
        prm.H = prm.E * prm.Et / (prm.E - prm.Et)
        prm.nu = 0.2
        prm.deg_d = 3
        prm.deg_q = 4

        law = plasticity_law(prm)
        problem = c.MechanicsProblem(mesh, prm, law=law)

        left = boundary.plane_at(0.0)
        right = boundary.plane_at(1.0)
        bc_expr = df.Expression(("u",), degree=0, u=0)

        bcs = []
        bcs.append(df.DirichletBC(problem.Vd, bc_expr, right))
        bcs.append(df.DirichletBC(problem.Vd, (0,), left))
        problem.set_bcs(bcs)
        
        ld = c.helper.LoadDisplacementCurve(bcs[0])

        # we do two load increments, the first to sigma0, the second to 
        # 2*sigma0
        eps_of_sig0 = prm.sig0/prm.E
        for t in np.linspace(0., 2 * eps_of_sig0 , 3):
            bc_expr.u = t
            problem.solve()
            problem.update()
            ld(t, df.assemble(problem.R))

        first_slope = (ld.load[1] - ld.load[0]) / eps_of_sig0
        second_slope = (ld.load[2] - ld.load[1]) / eps_of_sig0
        
        self.assertAlmostEqual(first_slope, prm.E)
        self.assertAlmostEqual(second_slope, prm.Et)

    def test_bending(self):
        # return
        LX = 6.0
        LY = 0.5
        LZ = 0.5
        mesh_resolution = 2.0

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
        law = plasticity_law(revise_prm(prm))

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
    # tests.test_1d()
    # tests.test_bending()
    # tests.test_at_one_point()
