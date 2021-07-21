import warnings

import dolfin as df
import numpy as np
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
from scipy.linalg import eigvals
import constitutive as c

df.parameters["form_compiler"]["representation"] = "quadrature"
warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)





mesh = df.UnitCubeMesh(1,1,1)
mesh0 = df.UnitCubeMesh(1,1,1)
V0 = df.VectorFunctionSpace(mesh0, "Lagrange", 1)
V = df.VectorFunctionSpace(mesh, "Lagrange", 1)
VD = df.VectorFunctionSpace(mesh,"DG",0, dim=6)


# Elasticity parameters
E, nu  = 42.0, 0.3
rho = 1e-1
t_end = 300
law = c.Hypoelasticity(E,nu)

all_ = df.CompiledSubDomain("true")
expr_all = df.Expression(("0.0", "u*x[0]", "0.0"), u=2*np.pi, degree=1)
bc_all = df.DirichletBC(V0, expr_all,all_)



expr_all.u = expr_all.u /t_end
bcs =[bc_all]

print("here 1")
def sym_grad_vector(u):
    e = df.sym(df.grad(u))
    return df.as_vector(
        [
            e[0, 0],
            e[1, 1],
            e[2, 2],
            2 ** 0.5 * e[1, 2],
            2 ** 0.5 * e[0, 2],
            2 ** 0.5 * e[0, 1],
        ]
    )
v_ = df.TestFunction(V)
u_ = df.TrialFunction(V)
print(law.C)

D = df.as_matrix(law.C.tolist())
K_form = df.inner(sym_grad_vector(u_), df.dot(D, sym_grad_vector(v_))) * df.dx
M_form = rho * df.inner(u_, v_) * df.dx

def critical_timestep(K_form, M_form, mesh):
    eig = 0.0
    for cell in df.cells(mesh):
        # get total element mass
        Me = df.assemble_local(M_form, cell)
        Ke = df.assemble_local(K_form, cell)
        eig = max(np.linalg.norm(eigvals(Ke, Me), np.inf), eig)
    
    h = 2.0 / eig ** 0.5
    return h

h = critical_timestep(K_form, M_form, mesh)


v = df.Function(V)
u = df.Function(V)

def lumped_mass(V, dx, rho):
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    mass_form = df.action(df.inner(u, v) * rho * dx, df.Constant((1.0, 1.0, 1.0)))
    return df.assemble(mass_form).get_local()

print("local dimension of V:", v.vector().get_local().size // 3)

solver = c.CDM(
    V, u, v, 0, None, bcs,  lumped_mass(V, df.dx, rho), law,  None
)

u_ = []
s_11 = []
s_12 = []
counter = 0

while np.any(solver.t < t_end):# and counter <= 2000:
    solver.step(h)
    if counter % 100 == 0:
        u_.append(expr_all.u * solver.t[0])
        s_11.append(solver.stress.vector().get_local()[0])
        s_12.append(solver.stress.vector().get_local()[5])
        print("t", solver.t[0], flush=True)

    counter += 1

import matplotlib.pyplot as plt
plt.plot(u_,s_11)
plt.plot(u_,s_12)
#plt.plot(u_,s_log)
plt.show()

