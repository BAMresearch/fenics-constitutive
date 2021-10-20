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

class PullTest(unittest.TestCase):
    def setUp(self):

        mesh = df.UnitCubeMesh(1,1,1)
        mesh0 = df.UnitCubeMesh(1,1,1)
        V0 = df.VectorFunctionSpace(mesh0, "Lagrange", 1)
        V = df.VectorFunctionSpace(mesh, "Lagrange", 1)
        VD = df.VectorFunctionSpace(mesh,"DG",0, dim=6)
        left = df.CompiledSubDomain("near(x[0], side) && on_boundary", side=0.0)
        right = df.CompiledSubDomain("near(x[0], side) && on_boundary", side=1.0)
        front = df.CompiledSubDomain("near(x[1],0.0) && on_boundary")
        back = df.CompiledSubDomain("near(x[2],0.0) && on_boundary")

        expr_left = df.Expression("0.0", degree=1)
        expr_right = df.Expression("u", u = 0.5, degree=1)
        expr_front = df.Expression("0.0", degree=1)
        expr_back = df.Expression("0.0", degree=1)


        # Elasticity parameters
        E, nu  = 42.0, 0.3
        self.E=E
        rho = 1e-1
        t_end = 150
        law = c.HookesLaw(E, nu, False, False)
        expr_right.u = expr_right.u / t_end
        print("here 1")
        bcl = df.DirichletBC(V0.sub(0), expr_left, left)
        bcr = df.DirichletBC(V0.sub(0), expr_right, right)
        bcf = df.DirichletBC(V.sub(1), expr_front, front)
        bcb = df.DirichletBC(V.sub(2), expr_back, back)
        bcs = [bcl, bcr, bcf, bcb]
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
        counter = 0

        while np.any(solver.t < t_end):# and counter <= 2000:
            solver.step(h)
            if counter % 1000 == 0:
                u_.append(expr_right.u * solver.t[0])
                s_11.append(solver.stress.vector().get_local()[0])
                print("t", solver.t[0]/t_end, flush=True)
            counter += 1
        self.u = np.array(u_)
        self.s11 = np.array(s_11)

    def test_log_stress(self):
        s_log = np.log(np.array(self.u)+1) * self.E
        np.testing.assert_array_almost_equal(s_log, self.s11, decimal=2)

if __name__ == "__main__":
    unittest.main()

