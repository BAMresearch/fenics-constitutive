import warnings

import dolfin as df
import numpy as np

from . import cpp, helper


class CDM:
    def __init__(
        self,
        V,
        u0,
        v0,
        t0,
        f_ext,
        bcs,
        M,
        law,
        stress_rate=None,
        damping_factor=None,
        calculate_F=False,
        bc_mesh="current",
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
        self.t = np.ones(self.QT.dim() // 9) * t0

        self.ip_loop = cpp.IpLoop()
        self.ip_loop.add_law(law, np.arange(self.QT.dim() // 9))
        self.ip_loop.resize(self.QT.dim() // 9)

        if stress_rate is not None:
            self.stress_rate = stress_rate
            self.stress_rate.resize(self.QT.dim() // 9)
        else:
            self.stress_rate = None

        self.f_ext = f_ext
        self.test_function = df.TestFunction(V)
        self.f_int_form = df.inner(
            helper.as_mandel(df.sym(df.grad(self.test_function))), self.stress
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

    def stress_update(self, h):

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
        if self.stress_rate is None:
            self.ip_loop.set(cpp.Q.SIGMA, self.stress.vector().get_local())
            self.ip_loop.set(cpp.Q.L, self.L.vector().get_local())
            self.ip_loop.set(cpp.Q.TIME_STEP, np.ones_like(self.t) * h)

            self.ip_loop.evaluate()
            self.ip_loop.update()
            new_stress = self.ip_loop.get(cpp.Q.SIGMA)
        else:
            self.stress_rate.set(self.L.vector().get_local())

            new_stress = self.stress.vector().get_local()
            new_stress = self.stress_rate(new_stress, h * 0.5)

            eps = cpp.strain_increment(self.L.vector().get_local(), h)

            self.ip_loop.set(cpp.Q.SIGMA, new_stress)
            self.ip_loop.set(cpp.Q.EPS, eps)

            self.ip_loop.evaluate()

            new_stress = self.ip_loop.get(cpp.Q.SIGMA)
            new_stress = self.stress_rate(new_stress, h * 0.5)

        helper.function_set(self.stress, new_stress)

    # @profile
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

        # if self.bcs is not None:
        # if self.bc_mesh == "current":
        # for bc in self.bcs:
        # bc.apply(self.v.vector())
        # elif self.bc_mesh == "initial":
        # helper.function_add(self.x, -self.u.vector().get_local())
        # df.set_coordinates(self.mesh.geometry(), self.x)
        # for bc in self.bcs:
        # bc.apply(self.v.vector())
        # helper.function_add(self.x, self.u.vector().get_local())

        if self.bcs is not None:
            for bc in self.bcs:
                bc.apply(self.v.vector())

        du = h * self.v.vector().get_local()
        helper.function_add(self.x, du * 0.5)

        df.set_coordinates(self.mesh.geometry(), self.x)

        self.stress_update(h)

        helper.function_add(self.u, du)
        helper.function_add(self.x, du * 0.5)
        df.set_coordinates(self.mesh.geometry(), self.x)

        self.t += h
