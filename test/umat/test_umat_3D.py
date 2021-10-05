import unittest
from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
import constitutive as c

set_log_active(False)

class TestUniaxial(unittest.TestCase):
    def test_3d_small(self):
        # ===== CREATE MESH =====
        mesh = Mesh()
        
        editor = MeshEditor()
        tdim = gdim = 3
        editor.open(mesh, 'tetrahedron', tdim, gdim)
        
        Lsize = 1.
        x = [0,Lsize]
        y = [0,Lsize]
        z = [0,Lsize]

        n = 2      # number of nodes in one dimension
        editor.init_vertices(n*n*n)

        vertex = 0
        for iz in range(n):
            for iy in range(n):
                for ix in range(n):
                    editor.add_vertex(vertex, [x[ix], y[iy], z[iz]])
                    vertex += 1

        editor.init_cells(6*(n-1)*(n-1)*(n-1))
        cell = 0
        for iz in range(n-1):
            for iy in range(n-1):
                for ix in range(n-1):
                    v0 = iz*n*n + iy*n + ix
                    v1 = v0 + 1;
                    v2 = v0 + n
                    v3 = v1 + n
                    v4 = v0 + n*n
                    v5 = v1 + n*n
                    v6 = v2 + n*n
                    v7 = v3 + n*n

                    editor.add_cell(cell, [v0, v1, v3, v7]);  cell += 1
                    editor.add_cell(cell, [v0, v1, v7, v5]);  cell += 1
                    editor.add_cell(cell, [v0, v5, v7, v4]);  cell += 1
                    editor.add_cell(cell, [v0, v3, v2, v7]);  cell += 1
                    editor.add_cell(cell, [v0, v6, v4, v7]);  cell += 1
                    editor.add_cell(cell, [v0, v2, v6, v7]);  cell += 1

        editor.close()

        # plot(mesh)
        # plt.show()

        # ===== Define Boundaries =====
        
        V = VectorFunctionSpace(mesh, 'CG', 1)
            
        def top(x, on_boundary):
            tol = 1e-14
            return (x[2] > Lsize - tol) and on_boundary
            
        def bottom(x, on_boundary):
            tol = 1e-14
            return x[2] < tol and on_boundary
            
        def origin(x, on_boundary):
            # vertex at the origin
            tol = 1e-14
            return x[0] < tol and x[1] < tol and x[2] < tol
        
        def pointX(x, on_boundary):
            # vertex on the edge along the x-axis
            tol = 1e-14
            return (x[0] > Lsize - tol) and x[1] < tol and x[2] < tol
            
        # ===== UMAT constitutive law =====
        constraint_type = c.Constraint.FULL
        prm = c.Parameters(constraint_type)
        prm.deg_d = 1

        law = c.Umat(constraint_type)
        #law = c.LinearElastic(1e9, 0.3, constraint_type)

        problem = c.MechanicsProblem(mesh, prm, law)
        
        # ===== Dirichlet BC =====
        load     = Expression("topDisplacement", topDisplacement=0.,degree = 1)
        bcLoad   = DirichletBC(problem.Vd.sub(2),load, top)
        bcBottom = DirichletBC(problem.Vd.sub(2), Constant(0.), bottom)
        bcOrigin = DirichletBC(problem.Vd, Constant((0.,0.,0.)), origin, method='pointwise')
        bcPointX = DirichletBC(problem.Vd.sub(1), Constant(0.), pointX, method='pointwise')
        
        bcs =[bcBottom, bcLoad, bcOrigin, bcPointX]
        
        problem.set_bcs(bcs)
        
        # ===== Output =====
        ld = c.helper.LoadDisplacementCurve(bcLoad)
        # ld.show()

        # fff = XDMFFile('test_umat3.xdmf')
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
        loadRate = 0.01*Lsize/T     # 1% within interval [0;T]
        
        #problem.update()
        # vtkfile = File('test_umat3.pvd')

        while (t < T + 1e-14):
            load.topDisplacement = loadRate*t
            
            law.updateTime(t_old,t)
            
            converged = solver.solve(problem, problem.u.vector())
            # assert converged

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
        # np.savetxt('LoadDisplCurve.txt',ld_array.T)
        # ld.keep()

        # ld_correct has been validated by Abaqus
        ld_correct = np.array([
            [0.0e+00, 0.0e+00, 0.00000000000e+00],
            [2.0e-03, 1.0e-03, 1.08138912533e+02],
            [4.0e-03, 2.0e-03, 1.97873150433e+02],
            [6.0e-03, 3.0e-03, 2.55116031682e+02],
            [8.0e-03, 4.0e-03, 2.85588489614e+02],
            [1.0e-02, 5.0e-03, 3.00941919349e+02],
            [1.2e-02, 6.0e-03, 3.09159140131e+02],
            [1.4e-02, 7.0e-03, 3.14243812368e+02],
            [1.6e-02, 8.0e-03, 3.17965638191e+02],
            [1.8e-02, 9.0e-03, 3.2106641932e+02],
            [2.0e-02, 1.0e-02, 3.2385147521e+02]])

        # compare the reaction forces
        for ind in np.arange(ld_correct.shape[0]):
            self.assertAlmostEqual(np.asarray(ld.load)[ind], ld_correct[ind][2])

        # remove the fort.7 file containing the parameters of
        # UMAT simulations
        import os
        if (os.path.isfile("fort.7")):
            os.remove("fort.7")

        print("test_3D_small ..... OK!")
                
if __name__ == "__main__":
    unittest.main()


