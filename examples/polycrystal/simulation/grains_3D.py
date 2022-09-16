import meshio
from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
import constitutive as c
from time import time
from pathlib import Path

# set_log_active(False)

def get_abaqus_times():
    increments = []
    with open("abaqusTimes.txt", "r") as f:
        line = f.readline()

        while True:
            if not line.strip():
                break

            values = line.split(" ")
            Line = []
            for value in values:
                if value.strip():
                    Line.append(value.strip())

            if len(Line) == 2 and Line[1] == 'toOutput':
                try:
                    increments.append([float(Line[0]),True])
                except:
                    print("***ERROR when reading abaqusTimes")
                    stop
            else:
                try:
                    increments.append([float(Line[0]),False])
                except:
                    print("***ERROR when reading abaqusTimes")
                    stop
    
            line = f.readline()

    return increments

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

# ===== READ ABAQUS INPUT FILE(S) =====
eval_time_start = time()
meshAba = meshio.read("grains.inp")

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
cell_domains = MeshFunction('size_t', mesh,mesh.topology().dim())
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
# define constraint type & interpolation order
constraint_type = c.Constraint.FULL
prm = c.Parameters(constraint_type)
prm.deg_d = 1   # if not given, quadrature order prm.deg_d = 2
prm.deg_q = 1

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
    law = c.Umat(
        constraint_type,
        "SDCRCRY",
        str(Path.home() / "CI_fenics" / "labtools-fenics" / "lib" / "libumat.so"),
        37,
        "kusdcrcry_",
        "param0_sdcrcry_",
        euler_angles,
    )
    iploop.add_law(law, np.where(ip_flags == grainID)[0])
    
    grainID += 1

# define problem, need a "template" law, which will be ommited
problem = c.MechanicsProblem_I(mesh, prm, law, iploop)

# ===== Define Boundaries =====
V = VectorFunctionSpace(mesh, 'CG', 1)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-14
        return (x[2] >  LmaxSizeZ - tol) and on_boundary
            
def bottom(x, on_boundary):
    tol = 1e-14
    return x[2] < LminSizeZ + tol and on_boundary
            
def fixed(x, on_boundary):
    # vertex keeps it position
    tol = 1e-14
    for node in pointsAba:
        if (near(node[2],LminSizeZ,tol)):
            nodeFix = node
            break
    return (near(x[0], nodeFix[0], tol) and near(x[1], nodeFix[1], tol)
            and x[2] < LminSizeZ + tol)

top = Top()

# ===== Dirichlet BC =====
bcBottom = DirichletBC(problem.Vd.sub(2), Constant(0.), bottom)
bcFixed  = DirichletBC(problem.Vd, Constant((0.,0.,0.)), fixed, method='pointwise')

bcs =[bcBottom]
problem.set_bcs(bcs)

# ===== add traction =====
# Define surface measure
facets = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
facets.set_all(0)
top.mark(facets, 1)

ds = Measure('ds', domain=mesh, subdomain_data=facets)

#====== vizualize BC =====
# u_ = Function(V)
# u_.vector()[:] = 0.5

# bcs =[bcLoad, bcFixed]
# [bc.apply(u_.vector()) for bc in bcs]

# File("u_.pvd") << u_
# exit()

# ===== Output =====
ld = c.helper.LoadDisplacementCurve(bcBottom)
# ld.show()

fff = XDMFFile('test_umat_grains_3D.xdmf')
fff.parameters["functions_share_mesh"] = True
fff.parameters["flush_output"] = True

import locale
locale.setlocale(locale.LC_NUMERIC, "en_US.UTF-8")
fff.write(cell_domains)

# ===== adjust solver =====
pc = PETScPreconditioner("petsc_amg")

# Use Chebyshev smoothing for multigrid
PETScOptions.set("mg_levels_ksp_type", "chebyshev")
PETScOptions.set("mg_levels_pc_type", "jacobi")

# Improve estimate of eigenvalues for Chebyshev smoothing
PETScOptions.set("mg_levels_esteig_ksp_type", "gmres")
PETScOptions.set("mg_levels_ksp_chebyshev_esteig_steps", 50)

lin_solver = PETScKrylovSolver("gmres", pc)
lin_solver.parameters["nonzero_initial_guess"] = True
lin_solver.parameters["maximum_iterations"] = 15000
lin_solver.parameters["relative_tolerance"] = 1.0e-8
lin_solver.parameters["error_on_nonconvergence"] = False

#linear_solver = LUSolver("mumps")
solver = NewtonSolver(
    MPI.comm_world, lin_solver, PETScFactory.instance())
#solver.parameters["linear_solver"] = "mumps"
solver.parameters["maximum_iterations"] = 10
solver.parameters["error_on_nonconvergence"] = False

# ===== solve over the time =====
# Step in time
t  = 0.0
t_old = -1.e-12
T = get_abaqus_times()

traction_old = 0.    # initiation of the traction
CreepLoad = 35.      # creep at 35 MPa
top_displ = []
        
V0 = FunctionSpace(mesh, "DG", 0)  # vizualization of stress

loop_time_start = time()
incr_time_start = loop_time_start
evalTime = []

u_old = Function(V)  # used for a better initial guess
u_rate = Function(V)

Dforce = Constant((0., 0., traction_old))
problem.add_force_term(dot(TestFunction(problem.Vd), Dforce)*ds(1))

for item in T:
    t = item[0]
    traction = (CreepLoad/10.)*(t if t <= 10. else 10.)
    
    Dforce.assign(Constant((0., 0., traction)))
    
    problem.iploop.updateTime(t_old,t)
    
    # initial guess
    problem.u.vector()[:] += (t - t_old) * u_rate.vector()[:]

    converged = solver.solve(problem, problem.u.vector())
    if converged[1]:
        print("...converged")
    # info(solver.parameters, True)
    if not converged[1]:
        print("...not converged")
    # assert converged
    
    problem.update()
    
    if item[1]:
        # this fixes XDMF time stamps
        import locale
        
        locale.setlocale(locale.LC_NUMERIC, "en_US.UTF-8")
        fff.write(problem.u, t)
        fff.write(project(problem.q_sigma[0], V0), t)
        
        ld(t, assemble(problem.R))
        top_displ.append(problem.u(0.780794969109, 0.086212993046, 1.)[2])  # node 340
        
    print("time step %4.3f" % t, " finished")
    
    traction_old = traction

    u_rate.vector()[:] = (problem.u.vector()[:] - u_old.vector()[:]) / (t - t_old)
    u_old.vector()[:] = problem.u.vector()[:]
    t_old = t

    incr_time_end = time()
    evalTime.append(incr_time_end - incr_time_start)
    print("...total time ", incr_time_end - eval_time_start)
    np.savetxt('grains.time',np.asarray([evalTime]).T)
    incr_time_start = incr_time_end

    # write at each increment
    ld_array = np.asarray([ld.ts,top_displ,ld.load])
    np.savetxt('LoadDisplCurve_grains.txt',ld_array.T)
# ld.keep()

# remove the fort.7 file containing the parameters of
# UMAT simulations
import os
if (os.path.isfile("fort.7")):
    os.remove("fort.7")
    
print("test_3D_grains ..... OK!")               



