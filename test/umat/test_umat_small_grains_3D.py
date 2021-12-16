import unittest
import meshio
from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
import constitutive as c

set_log_active(False)

def get_ip_flags(mesh, prm, mesh_function):
    # generic quadrature function spaces
    VQF, VQV, VQT = c.helper.spaces(mesh, prm.deg_q, c.q_dim(prm.constraint))

    # number of ip per cell
    q_eps = Function(VQV, name="current strains")
    n_gauss_points = len(q_eps.vector().get_local()) // c.q_dim(prm.constraint)
    nq = n_gauss_points // mesh.num_cells()

    # generate ip flags from mesh_function
    ip_flags = np.repeat(mesh_function.array(), nq)

    return ip_flags

class TestUniaxial(unittest.TestCase):
    def test_3d_small_grains(self):
        # ===== READ ABAQUS INPUT FILE(S) =====
        meshAba = meshio.read("3D_small_grains.inp")

        # if there is(are) an() element set(s) in Abaqus INP file,
        # which includs() all the elements, or must not belong to any section -
        # it(they) must be omitted;
        # provide this (these) set name(s) or use 'None' instead:
        # ElSetAllElements = ['None']
        ElSetAllElements = ['Eall']

        # get nodes
        pointsAba = meshAba.points

        # get elements of a particular type and element sets
        elType = "tetra"
        elementsAba = meshAba.get_cells_type(elType)

        cell_setsAba = meshAba.cell_sets_dict

        # ===== CREATE MESH =====
        mesh = Mesh()

        editor = MeshEditor()
        tdim = gdim = 3
        editor.open(mesh, 'tetrahedron', tdim, gdim)
        
        editor.init_vertices(pointsAba.shape[0])
        vertex = 0
        for node in pointsAba:
            editor.add_vertex(vertex,node)
            vertex += 1

        editor.init_cells(elementsAba.shape[0])
        cell = 0
        for element in elementsAba:
            editor.add_cell(cell, element)
            cell += 1

        editor.close()

        # plot(mesh)
        # plt.show()

        # introduce the grains into the mesh
        cell_domains = MeshFunction('size_t', mesh, mesh.topology().dim())
        cell_domains.set_all(0)

        grainID = 1
        for instance in cell_setsAba:
            cellIDs = cell_setsAba[instance][elType]
            if (instance in ElSetAllElements):
                continue
    
            for cellID in cellIDs:
                cell_domains[cellID] = grainID

            grainID += 1

        LminSizeZ = mesh.coordinates()[:,2].min()
        LmaxSizeZ = mesh.coordinates()[:,2].max()
        LmaxSizeX = mesh.coordinates()[:,0].max()

        # print(dir(cell_domains))
        # print(cell_domains.array())
        # print(mesh.cells())
        # print(cell_setsAba)
        
        # ===== read grain orientations =====
        ori = c.helper.OrientationFromAbaqus()
        ori.read_orientations_from("64_orientations.inp")
        ori.read_orientations_from("64_solidsections.inp")

        # check whether each grain (solid section) has a
        # model to be assigned
        if (ori.isComplete()):
            for instance in cell_setsAba:
                if (instance in ori.solid_section_data.keys(), ElSetAllElements):
                    continue
                else:
                    print("ERROR: any material law and orientation can be",
                    " assigned to the solid section ", instance)
                    exit()
        else:
            print("ERROR: computation of Euler angles was not complete.")
            exit()

        # ===== UMAT constitutive law =====
        constraint_type = c.Constraint.FULL
        prm = c.Parameters(constraint_type)
        prm.deg_d = 1   # if not given, quadrature order prm.deg_q = 2

        # get the array of all ips, with a marker from cell_domains
        ip_flags = get_ip_flags(mesh, prm, cell_domains)

        # create and fill IpLoop()
        iploop = c.IpLoop()
        grainID = 1
        for instance in cell_setsAba:
            if (instance in ElSetAllElements):
                continue
            law_name = ori.solid_section_data[instance][0]
            grain_orientation = ori.solid_section_data[instance][1]
            euler_angles = ori.orientation_data_Euler[grain_orientation]

            # create law and add to iploop, with the ids of ips, where the
            # law holds
            law = c.Umat(law_name, constraint_type, euler_angles)
            iploop.add_law(law, np.where(ip_flags == grainID)[0])

            grainID += 1

        # define problem, need a "template" law, which will be ommited
        problem = c.MechanicsProblem(mesh, prm, law, iploop)

        # ===== Define Boundaries =====
        V = VectorFunctionSpace(mesh, 'CG', 1)
            
        def top(x, on_boundary):
            tol = 1e-14
            return (x[2] >  LmaxSizeZ - tol) and on_boundary
            
        def bottom(x, on_boundary):
            tol = 1e-14
            return x[2] < LminSizeZ + tol and on_boundary
            
        def origin(x, on_boundary):
            # vertex at the origin
            tol = 1e-14
            return x[0] < tol and x[1] < tol and x[2] < LminSizeZ + tol
        
        def pointX(x, on_boundary):
            # vertex on the edge along the x-axis
            tol = 1e-14
            return (x[0] > LmaxSizeX - tol) and x[1] < tol and x[2] < LminSizeZ + tol
            
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

        # fff = XDMFFile('3D_small_grains.xdmf')
        # fff.parameters["functions_share_mesh"] = True
        # fff.parameters["flush_output"] = True
        # fff.write(cell_domains)    # does not work with time incrementation

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
        t_old = -1.e-12
        dt = 1.e-03
        T  = 20*dt
        loadRate = 0.01*(LmaxSizeZ - LminSizeZ)/T     # 1% within interval [0;T]
        
        V0 = FunctionSpace(mesh, "DG", 0)  # vizualization of stress

        while (t < T + 1e-14):
            load.topDisplacement = loadRate*t

            problem.iploop.updateTime(t_old,t)
            
            converged = solver.solve(problem, problem.u.vector())
            # assert converged

            problem.update()
            
            # this fixes XDMF time stamps
            import locale
            
            # locale.setlocale(locale.LC_NUMERIC, "en_US.UTF-8")
            # fff.write(problem.u, t)
            q_sigma_0 = project(problem.q_sigma[0], V0)
            q_sigma_0.rename("sigma_0", "sigma_0")
            fff.write(q_sigma_0, t)
                        
            ld(t, assemble(problem.R))
            
            print("time step %4.3f" % t, " finished")
            
            t_old = t
            t += dt

        ld_array = np.asarray([ld.ts,ld.disp,ld.load])
        # np.savetxt('LoadDisplCurve_3Dsmall_grains.txt',ld_array.T)
        # ld.keep()

        # ld_correct has been validated by Abaqus
        # Implicit analytic
        ld_correct = np.array([
            [0.0e+00, 0.0e+00, 0.00000000000e+00],
            [2.0e-03, 1.0e-03, 1.07810871252e+02],
            [4.0e-03, 2.0e-03, 1.58831473899e+02],
            [6.0e-03, 3.0e-03, 1.72401956707e+02],
            [8.0e-03, 4.0e-03, 1.75966754452e+02],
            [1.0e-02, 5.0e-03, 1.77205207354e+02],
            [1.2e-02, 6.0e-03, 1.77828994406e+02],
            [1.4e-02, 7.0e-03, 1.78248765082e+02],
            [1.6e-02, 8.0e-03, 1.78583151617e+02],
            [1.8e-02, 9.0e-03, 1.78874892417e+02],
            [2.0e-02, 1.0e-02, 1.79142674504e+02]])
        # Runge-Kutta
        #        ld_correct = np.array([
        #    [0.0e+00, 0.0e+00, 0.00000000000e+00],
        #    [2.0e-03, 1.0e-03, 1.12871287333e+02],
        #    [4.0e-03, 2.0e-03, 1.66033063486e+02],
        #    [6.0e-03, 3.0e-03, 1.74745520113e+02],
        #    [8.0e-03, 4.0e-03, 1.76635601301e+02],
        #    [1.0e-02, 5.0e-03, 1.77437734149e+02],
        #    [1.2e-02, 6.0e-03, 1.77934719363e+02],
        #    [1.4e-02, 7.0e-03, 1.78307551365e+02],
        #    [1.6e-02, 8.0e-03, 1.78619888437e+02],
        #    [1.8e-02, 9.0e-03, 1.78899415096e+02],
        #    [2.0e-02, 1.0e-02, 1.79159748167e+02]])

        # compare the reaction forces
        for ind in np.arange(ld_correct.shape[0]):
            self.assertAlmostEqual(np.asarray(ld.load)[2*ind], ld_correct[ind][2])

        # remove the fort.7 file containing the parameters of
        # UMAT simulations
        import os
        if (os.path.isfile("fort.7")):
            os.remove("fort.7")

        print("test_3D_small_grains ..... OK!")
                
if __name__ == "__main__":
    unittest.main()


