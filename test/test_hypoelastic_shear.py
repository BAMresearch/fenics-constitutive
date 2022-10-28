"""
WARNING: DOES NOT RUN WITH MPI
"""
import sys
import pandas as pd
import numpy as np
import dolfinx as dfx
import constitutiveX as cx
import basix
import ufl
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg", force=True)

from constitutiveX.helpers import QuadratureRule, get_local

from mpi4py import MPI

mesh = dfx.mesh.create_rectangle(MPI.COMM_WORLD,np.array([[0,0],[1., 1.]]), [1,1], cell_type=dfx.mesh.CellType.quadrilateral)
rule = QuadratureRule(quadrature_type=basix.QuadratureType.Default, cell_type=basix.CellType.quadrilateral, degree=1)

tdim = mesh.topology.dim
fdim = tdim - 1
mesh.topology.create_connectivity(fdim, tdim)

P1=dfx.fem.VectorFunctionSpace(mesh, ("CG", 1))
#V0=df.VectorFunctionSpace(mesh0, "CG", 1)
u = dfx.fem.Function(P1)
u_max= 5.
t_end = 100.
v_bc = u_max / t_end
domains = [
    lambda x : np.isclose(x[1],0.0), 
    lambda x : np.isclose(x[1],1.),
]
values=[(0.,0.),(v_bc,0.)]
#subspaces=[0,0,1,1]
#boundary_facets = [dfx.mesh.locate_entities_boundary(mesh, mesh.topology.dim-1, domain) for domain in domains]
#bc_dofs = [dfx.fem.locate_dofs_geometrical(P1, mesh.topology.dim-1, facet) for facet,i in zip(boundary_facets,subspaces)]
bc_dofs = [dfx.fem.locate_dofs_geometrical(P1,domain) for domain in domains]
#bc_dofs = [dfx.fem.locate_dofs_geometrical(P1, domain) for domain in domains]

bcs = [dfx.fem.dirichletbc(np.array(value), dofs, P1) for value,dofs in zip(values,bc_dofs)]

law = cx.HypoElastic3D(420.,0.3, rule.number_of_points(mesh))


#solution = pd.read_csv("CaseB.csv",header=0,decimal=",",sep =';')

v_ = ufl.TestFunction(P1)
u_ = ufl.TrialFunction(P1)


h = 1e-2 #c.critical_timestep(K_form, M_form, mesh)


mass_form = ufl.inner(u_, v_) * 42. * ufl.dx

#mass_action = dfx.fem.form(ufl.action(mass_form))

M = dfx.fem.petsc.assemble_matrix(dfx.fem.form(mass_form))
M.assemble()
ones = dfx.fem.Function(P1)
#ones = dfx.fem.Function(P1).vector
with ones.vector.localForm() as ones_local:
    ones_local.set(1.0)
M_action = M * ones.vector
#del M, ones


M_action.array[:] = 1./M_action.array
M_action.ghostUpdate()

#print("1")
solver = cx.CDMPlaneStrainX(
    P1, 0, None, bcs, M_action, law, rule
)

#print("1")
#s_eq_ = []
#p_ = []
#del_p = []
#damage =[]
u_ = []
s_11 = []
s_12 = []

counter = 0
while solver.t < t_end:# and counter <= 2000:
    solver.step(h)
    if counter % 100 == 0:
        u_.append(v_bc * solver.t)
        s_11.append(solver.stress.vector.array[0])
        s_12.append(solver.stress.vector.array[5])
        
    counter += 1

# if MPI.COMM_WORLD.rank==0:
plt.plot(u_,s_12)
#plt.scatter(solution.values[:, 0], solution.values[:, 1])
plt.title("JH2 Validation: Case C")
plt.xlabel("Pressure [GPa]")
plt.ylabel("Equiv. Stress [GPa]")
plt.savefig("pure_shear.png")

#from IPython import embed
#embed()
