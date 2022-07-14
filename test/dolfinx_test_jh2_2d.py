"""
WARNING: DOES NOT RUN WITH MPI
"""
import sys
import pandas as pd
import numpy as np
import dolfinx as dfx
import constitutive as c

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg", force=True)

from dolfinx_helpers import *
from mpi4py import MPI
def set_parameters(parameters,dic):
    for key, value in dic.items():
        parameters.__setattr__(key, value)

case1 = {"RHO": 3.7e-6,
         "SHEAR_MODULUS": 90.16,
         "A": 0.93, 
         "B": 0., 
         "C": 0., 
         "M": 0., 
         "N": 0.6, 
         "EPS0": 1.0, 
         "T": 0.2, 
         "SIGMAHEL": 2.0, 
         "PHEL": 1.46, 
         "D1": 0.0,
         "D2": 0.0,
         "K1": 130.95,
         "K2": 0.0,
         "K3": 0.0, 
         "BETA": 1.0,
         }

case2 = {"RHO": 3.7e-6,
         "SHEAR_MODULUS": 90.16,
         "A": 0.93, 
         "B": 0., 
         "C": 0., 
         "M": 0., 
         "N": 0.6, 
         "EPS0": 1.0, 
         "T": 0.2, 
         "SIGMAHEL": 2.0, 
         "PHEL": 1.46,# 38695149172, 
         "D1": 0.005,
         "D2": 1.0,
         "K1": 130.95,
         "K2": 0.0,
         "K3": 0.0, 
         "BETA": 1.0,
         }

case3 = {"RHO": 3.7e-6,
         "SHEAR_MODULUS": 90.16,
         "A": 0.93, 
         "B": 0.31, 
         "C": 0., 
         "M": 0.6, 
         "N": 0.6, 
         "EPS0": 1.0, 
         "T": 0.2, 
         "SIGMAHEL": 2.0, 
         "PHEL": 1.46, 
         "D1": 0.005,
         "D2": 1.0,
         "K1": 130.95,
         "K2": 0.0,
         "K3": 0.0, 
         "BETA": 1.0,
         }

# "one element test"

# print(solution.head())

mesh = dfx.mesh.create_rectangle(MPI.COMM_WORLD,np.array([[0,0],[1000., 1000.]]), [1,1], cell_type=dfx.mesh.CellType.quadrilateral)
rule = QuadratureRule(quadrature_type=basix.QuadratureType.Default, cell_type=basix.CellType.quadrilateral, degree=1)

tdim = mesh.topology.dim
fdim = tdim - 1
mesh.topology.create_connectivity(fdim, tdim)

P1=dfx.fem.VectorFunctionSpace(mesh, ("CG", 1))
#V0=df.VectorFunctionSpace(mesh0, "CG", 1)
u = dfx.fem.Function(P1)
t_end = 100.
v_bc = -50. / t_end
domains = [
    lambda x : np.isclose(x[0],0.0),
    lambda x : np.isclose(x[0],1000.), 
    lambda x : np.isclose(x[1],0.0), 
    lambda x : np.isclose(x[1],1000.),
]
values=[0.,0.,0., v_bc]
subspaces=[0,0,1,1]
boundary_facets = [dfx.mesh.locate_entities_boundary(mesh, mesh.topology.dim-1, domain) for domain in domains]
bc_dofs = [dfx.fem.locate_dofs_topological(P1.sub(i), mesh.topology.dim-1, facet) for facet,i in zip(boundary_facets,subspaces)]
#bc_dofs = [dfx.fem.locate_dofs_geometrical(P1, domain) for domain in domains]

bcs = [dfx.fem.dirichletbc(np.array(value), dofs, P1.sub(i)) for value,dofs,i in zip(values,bc_dofs,subspaces)]

parameters = c.JH2Parameters()
law = c.JH2Simple(parameters)

set_parameters(parameters, case3)

solution = pd.read_csv("CaseC.csv",header=0,decimal=",",sep =';')

v_ = ufl.TestFunction(P1)
u_ = ufl.TrialFunction(P1)


h = 1e-2 #c.critical_timestep(K_form, M_form, mesh)


mass_form = ufl.inner(u_, v_) * parameters.RHO * ufl.dx

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
solver = CDMPlaneStrain(
    P1, 0, None, bcs, M_action, law, rule
)

#print("1")
s_eq_ = []
p_ = []
del_p = []
damage =[]
counter = 0
total_mass = 1000.0**2 * parameters.RHO
while solver.t[0] < t_end:# and counter <= 2000:
    solver.step(h)
    if counter % 1 == 0:
        u_ = max(abs(get_local(solver.u)))

        density = total_mass / (1000 * (1000.-u_))
        print((density-law.get_internal_var(c.Q.RHO)[0])/density)
        s_mean =  np.mean(get_local(solver.stress).reshape((-1,6)), axis = 0)
        p = - (1/3) * np.sum(s_mean[:3])
        #print("###\n",get_local(solver.u))
        #print(get_local(solver.v),"###")
        s_dev = (s_mean + np.array([p,p,p,0.,0.,0.]))
        s_eq = (1.5 * np.inner(s_dev, s_dev))**0.5
        p_.append(p)
        s_eq_.append(s_eq)
        #damage.append(law.get_internal_var(c.Q.DAMAGE)[0])
        #del_p.append(law.get_internal_var(c.Q.PRESSURE)[0])
    counter += 1
# if MPI.COMM_WORLD.rank==0:
plt.plot(p_,s_eq_)
plt.scatter(solution.values[:, 0], solution.values[:, 1])
plt.title("JH2 Validation: Case C")
plt.xlabel("Pressure [GPa]")
plt.ylabel("Equiv. Stress [GPa]")
plt.savefig("jh2.png")
