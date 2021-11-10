import warnings
import unittest
import dolfin as df
import numpy as np
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
from scipy.linalg import eigvals
import constitutive as c

df.parameters["form_compiler"]["representation"] = "quadrature"
warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)


class ShearTest(unittest.TestCase):
    def setUp(self):

        mesh = df.UnitCubeMesh(2,2,2)
        mesh0 = df.UnitCubeMesh(2,2,2)
        V0 = df.VectorFunctionSpace(mesh0, "Lagrange", 1)
        V = df.VectorFunctionSpace(mesh, "Lagrange", 1)

        # Elasticity parameters
        E, nu  = 42.0, 0.3
        self.mu = E/(2*(1+nu))
        rho = 1e-1
        t_end = 10
        law = c.HookesLaw(E,nu,False,False)
        stress_rate = c.JaumannStressRate()
        all_ = df.CompiledSubDomain("true")
        expr_all = df.Expression(("0.0", "u*x[0]", "0.0"), u=2*np.pi, degree=1)
        bc_all = df.DirichletBC(V0, expr_all,all_)

        expr_all.u = expr_all.u /t_end
        bcs =[bc_all]

        v_ = df.TestFunction(V)
        u_ = df.TrialFunction(V)

        D = df.as_matrix(law.C.tolist())
        K_form = df.inner(c.as_mandel(df.sym(df.grad(u_))), df.dot(D, c.as_mandel(df.sym(df.grad(v_))))) * df.dx
        M_form = rho * df.inner(u_, v_) * df.dx

        h = c.critical_timestep(K_form, M_form, mesh)

        v = df.Function(V)
        u = df.Function(V)

        mass_form = df.action(df.inner(u_, v_) * rho * df.dx, df.Constant((1.0, 1.0, 1.0)))
        M = df.assemble(mass_form).get_local()


        solver = c.CDM(
            V, u, v, 0, None, bcs, M, law, stress_rate
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

    def test_s11(self):
        s11_exact = np.cos(self.u-np.pi)+1
        np.testing.assert_array_almost_equal(self.s11/self.mu, s11_exact, decimal = 2)

class ShearTestBC(unittest.TestCase):
    def setUp(self):

        mesh = df.UnitCubeMesh(2,2,2)
        #mesh0 = df.UnitCubeMesh(1,1,1)
        #V0 = df.VectorFunctionSpace(mesh0, "Lagrange", 1)
        V = df.VectorFunctionSpace(mesh, "Lagrange", 1)

        # Elasticity parameters
        E, nu  = 42.0, 0.3
        self.mu = E/(2*(1+nu))
        rho = 1e-1
        t_end = 10
        law = c.HookesLaw(E,nu,False,False)
        stress_rate = c.JaumannStressRate()
        all_ = df.CompiledSubDomain("true")
        expr_all = df.Expression(("0.0", "u*x[0]", "0.0"), u=2*np.pi, degree=1)
        bc_all = df.DirichletBC(V, expr_all,all_)

        expr_all.u = expr_all.u /t_end
        bcs =[bc_all]

        v_ = df.TestFunction(V)
        u_ = df.TrialFunction(V)

        D = df.as_matrix(law.C.tolist())
        K_form = df.inner(c.as_mandel(df.sym(df.grad(u_))), df.dot(D, c.as_mandel(df.sym(df.grad(v_))))) * df.dx
        M_form = rho * df.inner(u_, v_) * df.dx

        h = c.critical_timestep(K_form, M_form, mesh)

        v = df.Function(V)
        u = df.Function(V)

        mass_form = df.action(df.inner(u_, v_) * rho * df.dx, df.Constant((1.0, 1.0, 1.0)))
        M = df.assemble(mass_form).get_local()


        solver = c.CDM(
            V, u, v, 0, None, bcs, M, law, stress_rate,bc_mesh="initial"
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

    def test_s11(self):
        s11_exact = np.cos(self.u-np.pi)+1
        np.testing.assert_array_almost_equal(self.s11/self.mu, s11_exact, decimal = 2)

if __name__ == "__main__":
    unittest.main()
