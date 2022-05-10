import unittest
from fenics import *
from mshr import *
import numpy as np
import constitutive as c

from pathlib import Path
import subprocess

set_log_active(False)

libpath = "./libtestumat.so"


class TestUniaxial(unittest.TestCase):
    def setUp(self):
        """
        compile the fortran lib
        """
        umat_f = (
            Path(__file__).absolute().parent.parent / "src" / "umat_linear_elastic.f"
        )
        p = subprocess.run(
            ["gfortran", "-shared", "-fPIC", "-o", libpath, str(umat_f)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert not p.stderr

    def test_2d_notch(self):
        # ===== CREATE MESH =====
        mesh = Mesh()
        mesh_path = Path(__file__).absolute().parent / "umat" / "mesh2D.xdmf"
        with XDMFFile(str(mesh_path)) as f:
            f.read(mesh)

        LsizeY = mesh.coordinates()[:, :].max()

        # ===== Define Boundaries =====

        V = VectorFunctionSpace(mesh, "CG", 1)

        def top(x, on_boundary):
            tol = 1e-14
            return (x[1] > LsizeY - tol) and on_boundary

        def bottom(x, on_boundary):
            tol = 1e-14
            return x[1] < tol and on_boundary

        def origin(x, on_boundary):
            # vertex at the origin
            tol = 1e-14
            return x[0] < tol and x[1] < tol

        def sym_axis(x, on_boundary):
            tol = 1e-14
            return x[0] < tol and on_boundary

        # ===== UMAT constitutive law =====
        constraint_type = c.Constraint.PLANE_STRAIN
        prm = c.Parameters(constraint_type)
        prm.deg_d = 1
        prm.deg_q = 1

        law = c.Umat(
            constraint_type,
            "UMAT",
            libpath,
            0,
            "umat_",
            "",
            np.array([1e6, 0.3, 0.0]),
        )
        # law = c.LinearElastic(1e6, 0.3, constraint_type)

        problem = c.MechanicsProblem(mesh, prm, law)

        # ===== Dirichlet BC =====
        load = Expression("topDisplacement", topDisplacement=0.0, degree=1)
        bcLoad = DirichletBC(problem.Vd.sub(1), load, top)
        bcBottom = DirichletBC(problem.Vd.sub(1), Constant(0.0), bottom)
        bcOrigin = DirichletBC(
            problem.Vd, Constant((0.0, 0.0)), origin, method="pointwise"
        )
        bcSym = DirichletBC(problem.Vd.sub(0), Constant(0.0), sym_axis)

        bcs = [bcBottom, bcLoad, bcOrigin, bcSym]

        problem.set_bcs(bcs)

        # ===== Output =====
        ld = c.helper.LoadDisplacementCurve(bcLoad)
        # ld.show()

        fff = XDMFFile("test_umat2Dfile_linear_elastic.xdmf")
        fff.parameters["functions_share_mesh"] = True
        fff.parameters["flush_output"] = True

        # ===== adjust solver =====
        linear_solver = LUSolver("mumps")
        solver = NewtonSolver(MPI.comm_world, linear_solver, PETScFactory.instance())
        solver.parameters["linear_solver"] = "mumps"
        solver.parameters["maximum_iterations"] = 10
        solver.parameters["error_on_nonconvergence"] = False

        # ===== solve over the time =====
        # Step in time
        t = 0.0
        t_old = 0.0
        dt = 2.0e-03
        T = 10 * dt
        loadRate = 0.01 * LsizeY / T  # 1% within interval [0;T]

        V0 = FunctionSpace(mesh, "DG", 0)  # vizualization of stress & statevs
        nq = 1

        # problem.update()
        # vtkfile = File('test_umat2Dfile.pvd')

        while t < T + 1e-14:
            load.topDisplacement = loadRate * t

            law.updateTime(t_old, t)

            converged = solver.solve(problem, problem.u.vector())
            #    assert converged

            problem.update()

            # this fixes XDMF time stamps
            import locale

            locale.setlocale(locale.LC_NUMERIC, "en_US.UTF-8")
            fff.write(problem.u, t)

            q_sigma_0 = project(problem.q_sigma[0], V0)
            q_sigma_0.rename("sigma_0", "sigma_0")
            q_sigma_1 = project(problem.q_sigma[1], V0)
            q_sigma_1.rename("sigma_1", "sigma_1")
            q_sigma_2 = project(problem.q_sigma[2], V0)
            q_sigma_2.rename("sigma_2", "sigma_2")
            q_eps_0 = project(problem.q_eps[0], V0)
            q_eps_0.rename("eps_0", "eps_0")
            q_eps_1 = project(problem.q_eps[1], V0)
            q_eps_1.rename("eps_1", "eps_1")
            q_eps_2 = project(problem.q_eps[2], V0)
            q_eps_2.rename("eps_2", "eps_2")

            fff.write(q_sigma_0, t)
            fff.write(q_sigma_1, t)
            fff.write(q_sigma_2, t)
            fff.write(q_eps_0, t)
            fff.write(q_eps_1, t)
            fff.write(q_eps_2, t)

            # vtkfile << (problem.u, t)

            ld(t, assemble(problem.R))

            print("time step ", t, " finished")

            t_old = t
            t += dt

        ld_array = np.asarray([ld.ts, ld.disp, ld.load])
        # np.savetxt('LoadDisplCurve2DfileElastic.txt',ld_array.T)
        # ld.keep()

        # ld_correct has been validated by Abaqus
        ld_correct = np.array(
            [
                [0.0e00, 0.0e00, 0.00000000000e00],
                [2.0e-03, 1.5e-03, 7.67627325587e02],
                [4.0e-03, 3.0e-03, 1.53525465117e03],
                [6.0e-03, 4.5e-03, 2.30288197676e03],
                [8.0e-03, 6.0e-03, 3.07050930235e03],
                [1.0e-02, 7.5e-03, 3.83813662793e03],
                [1.2e-02, 9.0e-03, 4.60576395352e03],
                [1.4e-02, 1.05e-03, 5.37339127911e03],
                [1.6e-02, 1.2e-03, 6.14101860470e03],
                [1.8e-02, 1.35e-03, 6.90864593028e03],
                [2.0e-02, 1.5e-02, 7.67627325587e03],
            ]
        )

        # compare the reaction forces
        for ind in np.arange(ld_correct.shape[0]):
            self.assertAlmostEqual(np.asarray(ld.load)[ind], ld_correct[ind][2])

        # remove the fort.7 file containing the parameters of
        # UMAT simulations
        import os

        if os.path.isfile("fort.7"):
            os.remove("fort.7")

        print("test_2D_notch ..... OK!")


if __name__ == "__main__":
    unittest.main()
