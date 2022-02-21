"""
WARNING: DOES NOT RUN WITH MPI
"""
import pandas as pd
import numpy as np
import dolfin as df
import constitutive as c
import matplotlib.pyplot as plt

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

mesh = df.BoxMesh(
    df.Point(0.0, 0.0, 0.0), df.Point(1000.0, 1000.0, 1000.0), 1, 1, 1
)
mesh0 = df.BoxMesh(
    df.Point(0.0, 0.0, 0.0), df.Point(1000.0, 1000.0, 1000.0), 1, 1, 1
)
V=df.VectorFunctionSpace(mesh, "CG", 1)
V0=df.VectorFunctionSpace(mesh0, "CG", 1)

x_0 = df.CompiledSubDomain("near(x[0], 0.0) && on_boundary")
x_1 = df.CompiledSubDomain("near(x[0], 1000.0) && on_boundary")
y_0 = df.CompiledSubDomain("near(x[1], 0.0) && on_boundary")
y_1 = df.CompiledSubDomain("near(x[1], 1000.0) && on_boundary")
z_0 = df.CompiledSubDomain("near(x[2], 0.0) && on_boundary")
z_1 = df.CompiledSubDomain("near(x[2], 1000.0) && on_boundary")


expr_x_0 = df.Expression("0.0", degree=1)
expr_x_1 = df.Expression("0.0", degree=1)
expr_y_0 = df.Expression("0.0", degree=1)
expr_y_1 = df.Expression("0.0", degree=1)
expr_z_0 = df.Expression("0.0", degree=1)
expr_z_1 = df.Expression("u", u = 50, degree=1)

bc_x_0 = df.DirichletBC(V0.sub(0), 0., x_0)
bc_x_1 = df.DirichletBC(V0.sub(0), expr_x_1, x_1)
bc_y_0 = df.DirichletBC(V0.sub(1), expr_y_0, y_0)
bc_y_1 = df.DirichletBC(V0.sub(1), expr_y_1, y_1)
bc_z_0 = df.DirichletBC(V0.sub(2), expr_z_0, z_0)
bc_z_1 = df.DirichletBC(V0.sub(2), expr_z_1, z_1)
bcs = [bc_x_0, bc_x_1, bc_y_0, bc_y_1, bc_z_0, bc_z_1,]
parameters = c.JH2Parameters()
law = c.JH2Simple(parameters)

set_parameters(parameters, case3)
set_parameters(parameters, {"MOGEL": 1.})
set_parameters(parameters, {"D1": 0.00815})

solution = pd.read_csv("CaseC.csv",header=0,decimal=",",sep =';')

v_ = df.TestFunction(V)
u_ = df.TrialFunction(V)

# D = df.as_matrix(law.C.tolist())
# K_form = df.inner(c.as_mandel(df.sym(df.grad(u_))), df.dot(D, c.as_mandel(df.sym(df.grad(v_))))) * df.dx
# M_form = rho * df.inner(u_, v_) * df.dx

h = 1e-1 #c.critical_timestep(K_form, M_form, mesh)
t_end = 100.
expr_z_1.u = -50. / t_end

v = df.Function(V)
u = df.Function(V)

mass_form = df.action(df.inner(u_, v_) * parameters.RHO * df.dx, df.Constant((1.0, 1.0, 1.0)))
M = df.assemble(mass_form).get_local()

solver = c.CDM(
    V, u, v, 0, None, bcs, M, law, stress_rate = None 
)
print(solver.x.vector().get_local())

s_eq_ = []
p_ = []
del_p = []
damage =[]
counter = 0
total_mass = 1000.0**3 * parameters.RHO
while solver.t[0] < t_end:# and counter <= 2000:
    solver.step(h)
    if counter % 10 == 0:
        u_ = max(abs(solver.u.vector().get_local()))

        density = total_mass / (1000.**2 * (1000.-u_))
        # print((density-law.get_internal_var(c.Q.RHO)[0])/density)
        s_mean =  np.mean(solver.stress.vector().get_local().reshape((-1,6)), axis = 0)
        s_std =  np.std(solver.stress.vector().get_local().reshape((-1,6)), axis = 0)
        p = - (1/3) * np.sum(s_mean[:3])
        s_dev = (s_mean + np.array([p,p,p,0.,0.,0.]))
        s_eq = (1.5 * np.inner(s_dev, s_dev))**0.5
        p_.append(p)
        s_eq_.append(s_eq)
        damage.append(law.get_internal_var(c.Q.DAMAGE)[0])
        del_p.append(law.get_internal_var(c.Q.PRESSURE)[0])
    counter += 1


expr_z_1.u = -expr_z_1.u
while solver.t[0]< t_end*2:
    solver.step(h)
    if counter % 10 == 0:
        u_ = max(abs(solver.u.vector().get_local()))

        density = total_mass / (1000.**2 * (1000.-u_))
        # print((density-law.get_internal_var(c.Q.RHO)[0])/density)
        s_mean =  np.mean(solver.stress.vector().get_local().reshape((-1,6)), axis = 0)
        s_std =  np.std(solver.stress.vector().get_local().reshape((-1,6)), axis = 0)
        p = - (1/3) * np.sum(s_mean[:3])
        s_dev = (s_mean + np.array([p,p,p,0.,0.,0.]))
        s_eq = (1.5 * np.inner(s_dev, s_dev))**0.5
        p_.append(p)
        s_eq_.append(s_eq)
        damage.append(law.get_internal_var(c.Q.DAMAGE)[0])
        del_p.append(law.get_internal_var(c.Q.PRESSURE)[0])
    counter += 1
# print(s_eq_)
plt.plot(p_,s_eq_)
plt.scatter(solution.values[:, 0], solution.values[:, 1])
plt.title("JH2 Validation: Case C")
plt.xlabel("Pressure [GPa]")
plt.ylabel("Equiv. Stress [GPa]")
# plt.plot(p_,damage)
# plt.plot(p_,del_p)
plt.show()
