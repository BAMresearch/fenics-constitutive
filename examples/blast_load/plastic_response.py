"""
This example shall not use a stiffness matrix, but rather a calculation of internal forces.
Before: calculate matrix-vector product K*u (=f)
Now: Calculate f (=K*u) directly using ufl and assemble.

The Quadrature rule is for degree 1 and the mass matrix is lumped
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import constitutive as c
from scipy.interpolate import PchipInterpolator

import dolfin as df
"""
g 	mm 	ms 	N 	MPa 	N-mm
"""
try:
    #ht = 1e-4
    l_x = float(sys.argv[1])
    l_y = float(sys.argv[2])
    l_z = float(sys.argv[3])
    res = int(sys.argv[4])
    t_end = float(sys.argv[7])
except:
    if df.MPI.rank(df.MPI.comm_world):
        print("Couldn't read input, use default values instead")
    t_end = 110.0
    #ht = 1e-4
    l_x = 2500
    l_y = 2000
    l_z = 200
    res = 66


E = 27000.
sig0 = 20.
H = 400.
nu = 0.2
rho = 2000e-6


friedlander = pd.read_csv("Druckkurve_Beispiel.txt",header=0,decimal=",",sep ='\s+')

cs = PchipInterpolator(friedlander.values[:, 0], friedlander.values[:, 1])


def pressure(t):
    return -cs(t+2.4) * 1e-3 * 1.
    return -cs(t+2.4) * 1e-3

#plt.plot(np.linspace(0,50,100),-pressure(np.linspace(0,50,100)))
#plt.show()

mesh = df.BoxMesh(
    df.Point(0.0, 0.0, 0.0), df.Point(l_x, l_y, l_z), l_x//res, l_y//res, l_z//res
)
xdmf_file = df.XDMFFile(f"plastic_2_sides_size_{l_x}x{l_y}x{l_z}_res_{l_x//res}x{l_y//res}x{l_z//res}_{t_end}ms.xdmf")
#mesh_file = "random_plate_1"
#mesh = df.Mesh(mesh_file+".xml")
#xdmf_file = df.XDMFFile(f"plastic_2_sides_{mesh_file}_{t_end}ms_MPI.xdmf")


xdmf_file.parameters["flush_output"] = True
xdmf_file.parameters["functions_share_mesh"] = True
xdmf_file.parameters["rewrite_function_mesh"] = False

V = df.VectorFunctionSpace(mesh, "Lagrange", 1)
Q = c.quadrature_space(V)
V0 = df.FunctionSpace(mesh, "DG",0)
n = df.FacetNormal(mesh)

# u = TrialFunction(V)
v_ = df.TestFunction(V)
u_ = df.TrialFunction(V)

print("Dimension of V:", V.dim())


class Fixed(df.SubDomain):
    def inside(self, x, on_boundary):
        #return df.near(x[1], 0.0) and on_boundary
        return (df.near(x[0], 0.0) or df.near(x[0], l_x))  and df.near(x[2],0.0)#and on_boundary


class Exposed(df.SubDomain):
    def inside(self, x, on_boundary):
        return (
            df.near(x[2], 0.0) and on_boundary
        )  # and between(x[0], (40,60)) and between(x[1], (40,60))


fixed = Fixed()
exposed = Exposed()
boundaries = df.MeshFunction("size_t", mesh, mesh.geometric_dimension() - 1)
boundaries.set_all(0)
exposed.mark(boundaries, 2)


# hydro = Function(V, name = "Hydrostatic pressure")

ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)

pressure_bc = df.Expression(f"p", p=0.0, degree=3)

pressure_form = pressure_bc * df.dot(n, v_) * ds(2)


def forces(t):
    pressure_bc.p = pressure(t[0])
    return df.assemble(pressure_form)


bcs = [df.DirichletBC(V, (0.0, 0.0, 0.0), fixed, method="pointwise")]

law = c.AnalyticMisesPlasticity(E, nu, sig0, H, total_strains = False, tangent = False)
stress_rate = c.JaumannStressRate()
D = df.as_matrix(law.C.tolist())
K_form = df.inner(c.as_mandel(df.sym(df.grad(u_))), df.dot(D, c.as_mandel(df.sym(df.grad(v_))))) * df.dx
M_form = rho * df.inner(u_, v_) * df.dx

h = c.critical_timestep(K_form, M_form, mesh)

v = df.Function(V)
u = df.Function(V)

mass_form = df.action(df.inner(u_, v_) * rho * df.dx, df.Constant((1.0, 1.0, 1.0)))
M = df.assemble(mass_form).get_local()

solver = c.CDM(
    V, u, v, 0, forces, bcs, M, law, stress_rate
)
if df.MPI.rank(df.MPI.comm_world) == 0:
    print("critical time step:", h, "ms")


d = df.Function(V, name="Displacement")
q = df.Function(Q)
lam = df.Function(V0, name="plastic_strain")
dx = df.dx(
    metadata={
        "quadrature_degree": q.ufl_element().degree(),
        "quadrature_scheme": "default",
    }
)
xdmf_file.write(d,0.0)
xdmf_file.write(lam, solver.t[0])
# v = df.Function(V, name="Velocities")
# a = df.Function(V, name="Acceleration")
#@profile
def gogogo():
    # count = 0
    while np.any(solver.t < t_end):
        solver.step(h)
        # if count % 100 == 0:
            # d.assign(solver.u)
            # c.function_set(q,law.get_internal_var(c.Q.LAMBDA))
            # c.local_project(q,V0,dx,lam)
            # xdmf_file.write(d, solver.t[0])
            # xdmf_file.write(lam, solver.t[0])
            # if df.MPI.rank(df.MPI.comm_world) == 0:
                # print(solver.t[0], flush=True)
        # count += 1


gogogo()
if df.MPI.rank(df.MPI.comm_world) == 0:
    print("Simulation finished")
