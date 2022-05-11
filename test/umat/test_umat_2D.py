import unittest
from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
import constitutive as c
from time import time
from pathlib import Path

set_log_active(False)

class TestUniaxial(unittest.TestCase):
    def test_2d_small(self):
        # ===== CREATE MESH =====
        mesh = Mesh()

        editor = MeshEditor()
        tdim = gdim = 2
        editor.open(mesh, "triangle", tdim, gdim)

        Lsize = 1.0
        x = [0, 0.5 * Lsize, Lsize]
        y = [0, 0.5 * Lsize, Lsize]

        n = 3  # number of nodes in one dimension
        editor.init_vertices(n * n)

        vertex = 0
        for iy in range(n):
            for ix in range(n):
                editor.add_vertex(vertex, [x[ix], y[iy]])
                vertex += 1

        editor.init_cells(8)
        editor.add_cell(0, [0, 1, 3])
        editor.add_cell(1, [3, 1, 4])
        editor.add_cell(2, [4, 1, 2])
        editor.add_cell(3, [2, 5, 4])
        editor.add_cell(4, [6, 3, 4])
        editor.add_cell(5, [4, 7, 6])
        editor.add_cell(6, [7, 4, 5])
        editor.add_cell(7, [5, 8, 7])

        editor.close()

        # plot(mesh)
        # plt.show()

        # ===== Define Boundaries =====

        # V = VectorFunctionSpace(mesh, 'CG', 1)

        def top(x, on_boundary):
            tol = 1e-14
            return (x[1] > Lsize - tol) and on_boundary

        def bottom(x, on_boundary):
            tol = 1e-14
            return x[1] < tol and on_boundary

        def origin(x, on_boundary):
            # vertex at the origin
            tol = 1e-14
            return x[0] < tol and x[1] < tol

        # ===== UMAT constitutive law =====
        constraint_type = c.Constraint.PLANE_STRAIN
        prm = c.Parameters(constraint_type)
        prm.deg_d = 1
        prm.deg_q = 1

        law = c.Umat(
            constraint_type,
            "SDCHABOX",
            str(Path.home() / "Tools" / "labtools-fenics" / "lib" / "libumat.so"),
            29,
            "kusdchabox_",
            "param0_sdchabox_",
        ) 
        # return
        # law = c.LinearElastic(1e9, 0.3, constraint_type)

        problem = c.MechanicsProblem_I(mesh, prm, law)

        # ===== Dirichlet BC =====
        load = Expression("topDisplacement", topDisplacement=0.0, degree=1)
        bcLoad = DirichletBC(problem.Vd.sub(1), load, top)
        bcBottom = DirichletBC(problem.Vd.sub(1), Constant(0.0), bottom)
        bcOrigin = DirichletBC(
            problem.Vd, Constant((0.0, 0.0)), origin, method="pointwise"
        )

        bcs = [bcBottom, bcLoad, bcOrigin]

        problem.set_bcs(bcs)

        # ===== Output =====
        ld = c.helper.LoadDisplacementCurve(bcLoad)
        # ld.show()

        fff = XDMFFile("test_umat2Dsmall.xdmf")
        fff.parameters["functions_share_mesh"] = True
        fff.parameters["flush_output"] = True

        # ===== adjust solver =====
        pc = PETScPreconditioner("petsc_amg")

        # Use Chebyshev smoothing for multigrid
        PETScOptions.set("mg_levels_ksp_type", "chebyshev")
        PETScOptions.set("mg_levels_pc_type", "jacobi")

        # Improve estimate of eigenvalues for Chebyshev smoothing
        PETScOptions.set("mg_levels_esteig_ksp_type", "gmres")
        PETScOptions.set("mg_levels_ksp_chebyshev_esteig_steps", 50)

        lin_solver = PETScKrylovSolver("bicgstab", pc)
        lin_solver.parameters["nonzero_initial_guess"] = True
        lin_solver.parameters["maximum_iterations"] = 15000
        lin_solver.parameters["relative_tolerance"] = 1.0e-5
        lin_solver.parameters["error_on_nonconvergence"] = False

        # linear_solver = LUSolver("mumps")
        solver = NewtonSolver(MPI.comm_world, lin_solver, PETScFactory.instance())
        # solver.parameters["linear_solver"] = "mumps"
        solver.parameters["maximum_iterations"] = 10
        solver.parameters["error_on_nonconvergence"] = False

        # ===== solve over the time =====
        # Step in time
        t = 0.0
        t_old = 0.0
        dt = 2.0e-03
        T = 10 * dt
        loadRate = 0.01 * Lsize / T  # 1% within the interval [0;T]

        V0 = FunctionSpace(mesh, "DG", 0)  # vizualization of stress & statevs
        statev0 = Function(V0, name="statev0")
        statev6 = Function(V0, name="statev6")
        nq = 1

        # problem.update()
        # vtkfile = File('test_umat2Dsmall.pvd')

        while t < T + 1e-14:
            load.topDisplacement = loadRate * t

            law.updateTime(t_old, t)

            converged = solver.solve(problem, problem.u.vector())
            # assert converged
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
            statev0.vector().set_local(law.q_statev()[::29][::nq])
            statev6.vector().set_local(law.q_statev()[6::29][::nq])

            fff.write(q_sigma_0, t)
            fff.write(q_sigma_1, t)
            fff.write(q_sigma_2, t)
            fff.write(q_eps_0, t)
            fff.write(q_eps_1, t)
            fff.write(q_eps_2, t)
            fff.write(statev0, t)
            fff.write(statev6, t)

            #   vtkfile << (problem.u, t)

            ld(t, assemble(problem.R))

            print("time step ", t, " finished")

            t_old = t
            t += dt

        ld_array = np.asarray([ld.ts, ld.disp, ld.load])
        # np.savetxt('LoadDisplCurve2Dsmall.txt',ld_array.T)
        # ld.keep()

        # ld_correct has been validated by Abaqus
        ld_correct = np.array(
            [
                [0.0e00, 0.0e00, 0.00000000000e00],
                [2.0e-03, 1.5e-03, 1.19213570630e02],
                [4.0e-03, 3.0e-03, 2.16619858436e02],
                [6.0e-03, 4.5e-03, 2.78153158023e02],
                [8.0e-03, 6.0e-03, 3.11548774073e02],
                [1.0e-02, 7.5e-03, 3.29362554205e02],
                [1.2e-02, 9.0e-03, 3.39728667352e02],
                [1.4e-02, 1.05e-03, 3.46665123644e02],
                [1.6e-02, 1.2e-03, 3.51971992090e02],
                [1.8e-02, 1.35e-03, 3.56436159793e02],
                [2.0e-02, 1.5e-02, 3.60406088751e02],
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

        print("test_2D_small ..... OK!")

    def test_2d_notch(self):
        eval_time_start = time()
        # ===== CREATE MESH =====
        mesh = Mesh()
        with XDMFFile("mesh2D.xdmf") as f:
            f.read(mesh)

        LsizeY = mesh.coordinates()[:, :].max()

        # plot(mesh)
        # plt.show()

        # ===== Define Boundaries =====

        # V = VectorFunctionSpace(mesh, 'CG', 1)

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
        prm.deg_d = 2
        prm.deg_q = 2

        law = c.Umat(
            constraint_type,
            "SDCHABOX",
            str(Path.home() / "Tools" / "labtools-fenics" / "lib" / "libumat.so"),
            29,
            "kusdchabox_",
            "param0_sdchabox_",
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

        # ===== Get free DOFs =====
        # find out the dofs which are not attributed to any BCs
        # needed to print the residuum
        # bc_dofs = []
        # for bc in bcs:
        #     bc_dofs.extend(bc.get_boundary_values().keys())

        # dofs = [d for d in problem.Vd.dofmap().dofs() if d not in bc_dofs]

        # ===== Output =====
        ld = c.helper.LoadDisplacementCurve(bcLoad)
        # ld.show()

        fff = XDMFFile("test_umat2Dfile.xdmf")
        fff.parameters["functions_share_mesh"] = True
        fff.parameters["flush_output"] = True

        # ===== adjust solver =====
        linear_solver = LUSolver("mumps")
        solver = NewtonSolver(MPI.comm_world, linear_solver, PETScFactory.instance())
        solver.parameters["linear_solver"] = "mumps"
        solver.parameters["maximum_iterations"] = 10
        solver.parameters["relative_tolerance"] = 1e-10
        solver.parameters["error_on_nonconvergence"] = False
        # print(solver.parameters.keys())

        # ===== solve over the time =====
        # Step in time
        t = 0.0
        t_old = 0.0
        dt = 2.0e-03
        T = 10 * dt
        loadRate = 0.01 * LsizeY / T  # 1% within interval [0;T]

        # problem.update()
        # vtkfile = File('test_umat2Dfile.pvd')

        V0 = FunctionSpace(mesh, "DG", 0)  # vizualization of stress & statevs
        statev0 = Function(V0, name="statev0")
        statev6 = Function(V0, name="statev6")
        s0 = Function(V0, name="s0")
        s1 = Function(V0, name="s1")
        s2 = Function(V0, name="s2")
        # eps0 = Function(V0, name="e0")
        # eps1 = Function(V0, name="e1")
        # eqs2 = Function(V0, name="e2")

        nq = 3  # 3 for deg_q = 2 #1 for deg_q = 1 number of integration points per cell
        # print(len(law.q_statev()))
        # print(type(law.q_statev()))
        # print(dir(law.q_statev()))
        # print('#sigma#')
        # q_stress = problem.q_sigma.vector().get_local()
        # print(len(q_stress))
        # print(type(q_stress))
        # print(dir(q_stress))
        # print(len(q_stress[:: nq]))

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
            # statev0.vector().set_local(law.q_statev()[:: 29*nq])
            statev0.vector().set_local(law.q_statev()[::29][::nq])
            # statev6.vector().set_local(law.q_statev()[6 :: 29*nq])
            statev6.vector().set_local(law.q_statev()[6::29][::nq])

            q_stress = problem.q_sigma.vector().get_local()
            # q_eps = problem.q_eps.vector().get_local()
            s0.vector().set_local(q_stress[::3][::nq])
            s1.vector().set_local(q_stress[1::3][::nq])
            s2.vector().set_local(q_stress[2::3][::nq])
            # print(q_stress[:18])
            # print(q_eps[:18])
            # print(statev0.vector().get_local()[:6])
            # print(statev6.vector().get_local()[:6])

            # eps0.vector().set_local(problem.q_eps[0][:: nq])
            # eps1.vector().set_local(problem.q_eps[1][:: nq])
            # eps2.vector().set_local(problem.q_eps[2][:: nq])

            fff.write(q_sigma_0, t)
            fff.write(q_sigma_1, t)
            fff.write(q_sigma_2, t)
            fff.write(q_eps_0, t)
            fff.write(q_eps_1, t)
            fff.write(q_eps_2, t)
            fff.write(statev0, t)
            fff.write(statev6, t)

            fff.write(s0, t)
            fff.write(s1, t)
            fff.write(s2, t)
            # fff.write(eps0, t)
            # fff.write(eps1, t)
            # fff.write(eps2, t)

            # vtkfile << (problem.u, t)

            ld(t, assemble(problem.R))

            print("time step ", t, " finished")
            # here take only dofs, which are not attributed to any bcs, see helper for ld
            # print(np.absolute(assemble(problem.R).get_local()[dofs]).max())

            t_old = t
            t += dt

        eval_time_end = time()
        ld_array = np.asarray([ld.ts, ld.disp, ld.load])
        # np.savetxt('LoadDisplCurve2Dfile.txt',ld_array.T)
        print("Elapsed time {}".format(eval_time_end - eval_time_start))
        # with open('LoadDisplCurve2Dfile.txt', 'a') as f:
        #    f.write('Elapsed time {}'.format(eval_time_end - eval_time_start))
        # ld.keep()

        # ld_correct has been validated by Abaqus
        ld_correct = np.array(
            [
                [0.0e00, 0.0e00, 0.00000000000e00],
                [2.0e-03, 1.5e-03, 8.25768896852e01],
                [4.0e-03, 3.0e-03, 1.49266529477e02],
                [6.0e-03, 4.5e-03, 1.91717795041e02],
                [8.0e-03, 6.0e-03, 2.15659692494e02],
                [1.0e-02, 7.5e-03, 2.29115128242e02],
                [1.2e-02, 9.0e-03, 2.37256148289e02],
                [1.4e-02, 1.05e-03, 2.42749678642e02],
                [1.6e-02, 1.2e-03, 2.46876796586e02],
                [1.8e-02, 1.35e-03, 2.50247807184e02],
                [2.0e-02, 1.5e-02, 2.53159912618e02],
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
