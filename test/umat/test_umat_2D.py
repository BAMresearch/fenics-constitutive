import unittest
from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
import constitutive as c

set_log_active(False)

class TestUniaxial(unittest.TestCase):
    def test_2d_small(self):
        # ===== CREATE MESH =====
        mesh = Mesh()

        editor = MeshEditor()
        tdim = gdim = 2
        editor.open(mesh, 'triangle', tdim, gdim)

        Lsize = 1.
        x = [0, 0.5*Lsize, Lsize]
        y = [0, 0.5*Lsize, Lsize]

        n = 3      # number of nodes in one dimension
        editor.init_vertices(n*n)

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

        V = VectorFunctionSpace(mesh, 'CG', 1)

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

        law = c.Umat(constraint_type)
        #law = c.LinearElastic(1e9, 0.3, constraint_type)
        
        problem = c.MechanicsProblem(mesh, prm, law)

        # ===== Dirichlet BC =====
        load     = Expression("topDisplacement", topDisplacement=0.,degree = 1)
        bcLoad   = DirichletBC(problem.Vd.sub(1),load, top)
        bcBottom = DirichletBC(problem.Vd.sub(1), Constant(0.), bottom)
        bcOrigin = DirichletBC(problem.Vd, Constant((0.,0.)), origin, method='pointwise')

        bcs =[bcBottom, bcLoad, bcOrigin]

        problem.set_bcs(bcs)

        # ===== Output =====
        ld = c.helper.LoadDisplacementCurve(bcLoad)
        # ld.show()
        
     #   fff = XDMFFile('test_umat2Dsmall.xdmf')
     #   fff.parameters["functions_share_mesh"] = True
     #   fff.parameters["flush_output"] = True

        # ===== adjust solver =====
        linear_solver = LUSolver("mumps")
        solver = NewtonSolver(
            MPI.comm_world, linear_solver, PETScFactory.instance())
        solver.parameters["linear_solver"] = "mumps"
        solver.parameters["maximum_iterations"] = 10
        solver.parameters["error_on_nonconvergence"] = False

        # ===== solve over the time =====
        # Step in time
        t  = 0.0
        t_old = 0.0
        dt = 2.e-03
        T  = 10*dt
        loadRate = 0.01*Lsize/T     # 1% within the interval [0;T]

        #problem.update()
        # vtkfile = File('test_umat2Dsmall.pvd')
        
        while (t < T + 1e-14):
            load.topDisplacement = loadRate*t

            law.updateTime(t_old,t)
            
            converged = solver.solve(problem, problem.u.vector())
            # assert converged
            problem.update()
            
            # this fixes XDMF time stamps
         #   import locale
            
         #   locale.setlocale(locale.LC_NUMERIC, "en_US.UTF-8")
         #   fff.write(problem.u, t)
            
         #   vtkfile << (problem.u, t)
            
            ld(t, assemble(problem.R))
            
            print("time step ", t, " finished")
            
            t_old = t
            t += dt
            
        ld_array = np.asarray([ld.ts,ld.disp,ld.load])
        # np.savetxt('LoadDisplCurve2Dsmall.txt',ld_array.T)
        # ld.keep()

        # ld_correct has been validated by Abaqus
        ld_correct = np.array([
            [0.0e+00, 0.0e+00, 0.00000000000e+00],
            [2.0e-03, 1.0e-03, 1.20182533771e+02],
            [4.0e-03, 2.0e-03, 2.22701678732e+02],
            [6.0e-03, 3.0e-03, 2.91904635390e+02],
            [8.0e-03, 4.0e-03, 3.31470023347e+02],
            [1.0e-02, 5.0e-03, 3.52951291795e+02],
            [1.2e-02, 6.0e-03, 3.65180994836e+02],
            [1.4e-02, 7.0e-03, 3.72965682189e+02],
            [1.6e-02, 8.0e-03, 3.78605552127e+02],
            [1.8e-02, 9.0e-03, 3.83155704801e+02],
            [2.0e-02, 1.0e-02, 3.87099472437e+02]])

        # compare the reaction forces
        for ind in np.arange(ld_correct.shape[0]):
            self.assertAlmostEqual(np.asarray(ld.load)[ind], ld_correct[ind][2])

        # remove the fort.7 file containing the parameters of
        # UMAT simulations
        import os
        if (os.path.isfile("fort.7")):
            os.remove("fort.7")

        print("test_2D_small ..... OK!")

    def test_2d_notch(self):
        # ===== CREATE MESH =====
        mesh = Mesh()
        with XDMFFile("mesh2D.xdmf") as f:
            f.read(mesh)

        LsizeY = mesh.coordinates()[:,:].max()

        # plot(mesh)
        # plt.show()

        # ===== Define Boundaries =====

        V = VectorFunctionSpace(mesh, 'CG', 1)

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

        law = c.Umat(constraint_type)
        #law = c.LinearElastic(1e6, 0.3, constraint_type)

        problem = c.MechanicsProblem(mesh, prm, law)

        # ===== Dirichlet BC =====
        load     = Expression("topDisplacement", topDisplacement=0.,degree = 1)
        bcLoad   = DirichletBC(problem.Vd.sub(1),load, top)
        bcBottom = DirichletBC(problem.Vd.sub(1), Constant(0.), bottom)
        bcOrigin = DirichletBC(problem.Vd, Constant((0.,0.)), origin, method='pointwise')
        bcSym    = DirichletBC(problem.Vd.sub(0), Constant(0.), sym_axis)

        bcs =[bcBottom, bcLoad, bcOrigin, bcSym]

        problem.set_bcs(bcs)

        # ===== Output =====
        ld = c.helper.LoadDisplacementCurve(bcLoad)
        # ld.show()

        # fff = XDMFFile('test_umat2Dfile.xdmf')
        # fff.parameters["functions_share_mesh"] = True
        # fff.parameters["flush_output"] = True

        # ===== adjust solver =====
        linear_solver = LUSolver("mumps")
        solver = NewtonSolver(
            MPI.comm_world, linear_solver, PETScFactory.instance())
        solver.parameters["linear_solver"] = "mumps"
        solver.parameters["maximum_iterations"] = 10
        solver.parameters["error_on_nonconvergence"] = False

        # ===== solve over the time =====
        # Step in time
        t  = 0.0
        t_old = 0.0
        dt = 2.e-03
        T  = 10*dt
        loadRate = 0.01*LsizeY/T     # 1% within interval [0;T]

        # problem.update()
        # vtkfile = File('test_umat2Dfile.pvd')


        while (t < T + 1e-14):
            load.topDisplacement = loadRate*t

            law.updateTime(t_old,t)

            converged = solver.solve(problem, problem.u.vector())
            #    assert converged

            problem.update()

            # this fixes XDMF time stamps
            # import locale

            # locale.setlocale(locale.LC_NUMERIC, "en_US.UTF-8")
            # fff.write(problem.u, t)

            # vtkfile << (problem.u, t)
    
            ld(t, assemble(problem.R))

            print("time step ", t, " finished")
    
            t_old = t
            t += dt

        ld_array = np.asarray([ld.ts,ld.disp,ld.load])
        # np.savetxt('LoadDisplCurve2Dfile.txt',ld_array.T)
        # ld.keep()
        
        # ld_correct has been validated by Abaqus
        ld_correct = np.array([
            [0.0e+00, 0.0e+00,  0.00000000000e+00],
            [2.0e-03, 1.5e-03,  8.37150224145e+01],
            [4.0e-03, 3.0e-03,  1.54268025457e+02],
            [6.0e-03, 4.5e-03,  2.02083738840e+02],
            [8.0e-03, 6.0e-03,  2.30467259194e+02],
            [1.0e-02, 7.5e-03,  2.46860394564e+02],
            [1.2e-02, 9.0e-03,  2.56799361099e+02],
            [1.4e-02, 1.05e-03, 2.63392587707e+02],
            [1.6e-02, 1.2e-03,  2.68223389968e+02],
            [1.8e-02, 1.35e-03, 2.72079427521e+02],
            [2.0e-02, 1.5e-02,  2.75356456181e+02]])

        # compare the reaction forces
        for ind in np.arange(ld_correct.shape[0]):
            self.assertAlmostEqual(np.asarray(ld.load)[ind], ld_correct[ind][2])

        # remove the fort.7 file containing the parameters of
        # UMAT simulations
        import os
        if (os.path.isfile("fort.7")):
            os.remove("fort.7")

        print("test_2D_notch ..... OK!")
                
if __name__ == "__main__":
    unittest.main()


