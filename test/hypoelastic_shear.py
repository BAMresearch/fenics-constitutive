import warnings
import unittest
import dolfin as df
import numpy as np
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
from scipy.linalg import eigvals
import constitutive as c

df.parameters["form_compiler"]["representation"] = "quadrature"
warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)

def critical_timestep(K_form, M_form, mesh):
    eig = 0.0
    for cell in df.cells(mesh):
        # get total element mass
        Me = df.assemble_local(M_form, cell)
        Ke = df.assemble_local(K_form, cell)
        eig = max(np.linalg.norm(eigvals(Ke, Me), np.inf), eig)

    h = 2.0 / eig ** 0.5
    return h

class ShearTest(unittest.TestCase):
    def setUp(self):

        mesh = df.UnitCubeMesh(1,1,1)
        mesh0 = df.UnitCubeMesh(1,1,1)
        V0 = df.VectorFunctionSpace(mesh0, "Lagrange", 1)
        V = df.VectorFunctionSpace(mesh, "Lagrange", 1)

        # Elasticity parameters
        E, nu  = 42.0, 0.3
        self.mu = E/(2*(1+nu))
        rho = 1e-1
        t_end = 10
        law = c.HookesLaw(E,nu,False,False)
        all_ = df.CompiledSubDomain("true")
        expr_all = df.Expression(("0.0", "u*x[0]", "0.0"), u=2*np.pi, degree=1)
        bc_all = df.DirichletBC(V0, expr_all,all_)

        expr_all.u = expr_all.u /t_end
        bcs =[bc_all]

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

        D = df.as_matrix(law.C.tolist())
        K_form = df.inner(sym_grad_vector(u_), df.dot(D, sym_grad_vector(v_))) * df.dx
        M_form = rho * df.inner(u_, v_) * df.dx

        h = critical_timestep(K_form, M_form, mesh)

        v = df.Function(V)
        u = df.Function(V)

        mass_form = df.action(df.inner(u_, v_) * rho * df.dx, df.Constant((1.0, 1.0, 1.0)))
        M = df.assemble(mass_form).get_local()


        solver = c.CDM2(
            V, u, v, 0, None, bcs, M, law,  None
        )

        u_ = []
        s_11 = []
        s_12 = []
        counter = 0

        while np.any(solver.t < t_end):# and counter <= 2000:
            solver.step(h*0.5)
            if counter % 10 == 0:
                u_.append(expr_all.u * solver.t[0])
                s_11.append(solver.stress.vector().get_local()[0])
                s_12.append(solver.stress.vector().get_local()[5])

            counter += 1

        self.u = np.array(u_)
        self.s11 = np.array(s_11)
        self.s12 = np.array(s_12)

    def test_s12(self):
        s12_exact = np.sin(self.u)
        np.testing.assert_array_almost_equal(self.s12/2**0.5/self.mu, s12_exact, decimal = 2)
        #error = np.abs(self.s12/2**0.5/self.mu-s12_exact)
        #print(max(error))
        #print(error)
        #import matplotlib.pyplot as plt
        #plt.plot(self.u, error)
        #plt.yscale("log")
        #plt.show()

    def test_s11(self):
        s11_exact = np.cos(self.u-np.pi)+1
        np.testing.assert_array_almost_equal(self.s11/self.mu, s11_exact, decimal = 2)
# import matplotlib.pyplot as plt
# mu = E/(2*(1+nu))
# u_ = np.array(u_)
# plt.plot(u_,np.array(s_11)/mu)
# plt.plot(u_,np.array(s_12)/2**0.5/mu)
# plt.plot(u_,np.cos(u_-np.pi)+1)
# plt.plot(u_,np.sin(u_))
#plt.plot(u_,s_log)
# plt.show()

# plt.plot(u_,np.abs(np.array(s_11)/mu -np.cos(u_-np.pi)-1))
# plt.plot(u_,np.abs(np.array(s_12)/2**0.5/mu-np.sin(u_)))
# plt.yscale("log")
# plt.show()
if __name__ == "__main__":
    unittest.main()
