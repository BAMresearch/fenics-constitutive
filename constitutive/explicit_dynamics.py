import warnings

import numpy as np

import dolfin as df
from . import helper as h
import .cpp as cpp


class CDM:
    def __init__(
        self, V, u0, v0, t0, f_ext, bcs, M, damping_factor=None, calculate_F = False
    ):
        self.QT = h.quadrature_tensor_space(V, shape=(3, 3))
        self.QV = h.quadrature_vector_space(V, dim=6)
        # self.Q = quadrature_space(V)

        self.stress = df.Function(self.QV)
        self.mesh = V.mesh()
        self.v = v0
        self.u = u0
        self.a = np.zeros_like(self.u.vector().get_local())
        self.L = df.Function(self.QT)
        self.F = df.Function(self.QT) if calculate_F else None
        self.t = np.ones(self.a.size//3) * t0
        self.ip_loop = cpp.IpLoop()
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
        # self.calculate_F = calculate_F

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
        function_set(self.v, c1 * self.v.vector().get_local() + c2 * h * self.a)
        # self.v.vector().set_local(c1 * self.v.vector().get_local() + c2 * h * self.a)
        # self.v.vector().apply("insert")
        if self.bcs is not None:
            for bc in self.bcs:
                bc.apply(self.v.vector())

        # x_old = self.x.vector().get_local()
        # u_old = self.u.vector().get_local()

        du = h * self.v.vector().get_local()
        function_add(self.x, du * 0.5)
        # self.x.vector().add_local(du * 0.5)
        # self.x.vector().apply("insert")

        # we don't need to apply the boundary condition to x, if we apply bc to velocities

        df.set_coordinates(self.mesh.geometry(), self.x)

        # This can hopefully be done more efficiently

        local_project(
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

        self.ip_loop.set(cpp.SIGMA, self.stress.vector().get_local())
        self.ip_loop.set(cpp.L, self.L.vector().get_local())
        self.ip_loop.set(cpp.TIME_STEP, np.ones_like(self.t) * h)
        self.ip_loop.evaluate()
        function_set(self.stress, self.ip_loop.get(cpp.SIGMA))
        # self.stress.vector().set_local(new_stress)
        # self.stress.vector().apply("insert")
        # calculate internal forces at t_n+1, therefore we first need to set nodal positions

        function_add(self.u, du)
        function_add(self.x, du * 0.5)

        # self.u.vector().add_local(du)
        # self.x.vector().add_local(du * 0.5)
        # self.u.vector().apply("insert")
        # self.x.vector().apply("insert")

        # df.MPI.barrier(df.MPI.comm_world)
        df.set_coordinates(self.mesh.geometry(), self.x)

        self.t += h
