import warnings

import numpy as np

import dolfin as df
from . import helper
from . import cpp

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
class CDM:
    def __init__(
        self, V, u0, v0, t0, f_ext, bcs, M, law, damping_factor=None, calculate_F = False
    ):
        self.QT = helper.quadrature_tensor_space(V, shape=(3, 3))
        self.QV = helper.quadrature_vector_space(V, dim=6)

        self.stress = df.Function(self.QV)
        self.mesh = V.mesh()
        self.v = v0
        self.u = u0
        self.a = np.zeros_like(self.u.vector().get_local())
        self.L = df.Function(self.QT)
        self.F = df.Function(self.QT) if calculate_F else None
        self.t = np.ones(self.QT.dim()//9) * t0
        
        self.ip_loop = cpp.IpLoop()
        self.ip_loop.add_law(law, np.arange(self.QT.dim()//9))
        self.ip_loop.resize(self.QT.dim()//9)
        
        self.f_ext = f_ext
        self.test_function = df.TestFunction(V)
        self.f_int_form = df.inner(
            sym_grad_vector(self.test_function), self.stress
        ) * df.dx(
            metadata={
                "quadrature_degree": self.stress.ufl_element().degree(),
                "quadrature_scheme": "default",
            }
        )

        self.bcs = bcs
        self.M_inv = 1 / M
        self.x = df.interpolate(df.Expression(("x[0]", "x[1]", "x[2]"), degree=1), V)
        self.damping_factor = damping_factor
    
    def step(self, h):

        if self.damping_factor is not None:
            c = (h * self.damping_factor) / 2
            c1 = (1 - c) / (1 + c)
            c2 = 1 / (1 + c)
        else:
            c1, c2 = 1, 1

        f_int = df.assemble(self.f_int_form)
        f = -f_int + self.f_ext(self.t) if self.f_ext is not None else -f_int
        # calculate accelerations
        self.a = self.M_inv * f.get_local()
        # given: v_n-1/2, x_n/u_n, a_n, f_int_n
        # Advance velocities and nodal positions in time
        # multiply with damping factors if needed
        helper.function_set(self.v, c1 * self.v.vector().get_local() + c2 * h * self.a)
        if self.bcs is not None:
            for bc in self.bcs:
                bc.apply(self.v.vector())

        du = h * self.v.vector().get_local()
        helper.function_add(self.x, du * 0.5)

        df.set_coordinates(self.mesh.geometry(), self.x)

        helper.local_project(
            df.nabla_grad(self.v),
            self.QT,
            df.dx(
                metadata={
                    "quadrature_degree": self.stress.ufl_element().degree(),
                    "quadrature_scheme": "default",
                }
            ),
            self.L,
        )
        self.ip_loop.set(cpp.Q.SIGMA, self.stress.vector().get_local())
        self.ip_loop.set(cpp.Q.L, self.L.vector().get_local())
        self.ip_loop.set(cpp.Q.TIME_STEP, np.ones_like(self.t) * h)
        
        self.ip_loop.evaluate()

        helper.function_set(self.stress, self.ip_loop.get(cpp.Q.SIGMA))

        helper.function_add(self.u, du)
        helper.function_add(self.x, du * 0.5)

        df.set_coordinates(self.mesh.geometry(), self.x)

        self.t += h
