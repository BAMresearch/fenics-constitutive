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
        update_mesh = True,
    ):

        self.mesh = V.mesh()
        self.v = v0
        self.u = u0
        self.a = np.zeros_like(self.u.vector().get_local())

        self.QT = helper.quadrature_tensor_space(V, shape=(3, 3))
        self.QV = helper.quadrature_vector_space(V, dim=6)

        self.L = df.Function(self.QT)
        self.F = df.Function(self.QT) if calculate_F else None

        self.stress = df.Function(self.QV)

        self.local_q_dim = self.stress.vector().get_local().size // 6

        self.t = np.ones(self.local_q_dim) * t0

        self.ip_loop = cpp.IpLoop()
        self.ip_loop.add_law(law, np.arange(self.local_q_dim))
        self.ip_loop.resize(self.local_q_dim)

        
        self.stress_rate = stress_rate
        if stress_rate is not None:
            self.stress_rate.resize(self.local_q_dim)

        self.f_ext = f_ext

        test_function = df.TestFunction(V)

        self.f_int_form = df.inner(
            helper.as_mandel(df.sym(df.grad(test_function))), self.stress
        ) * df.dx(
            metadata={
                "quadrature_degree": self.stress.ufl_element().degree(),
                "quadrature_scheme": "default",
            }
        )

        self.bcs = bcs
        self.bc_mesh = bc_mesh
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

        self.ip_loop.update()
        helper.function_set(self.stress, new_stress)

    #@profile
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
        if self.bc_mesh == "current":
            for bc in self.bcs:
                bc.apply(self.v.vector())
        elif self.bc_mesh == "initial":
            u_vec = helper.function_get(self.u)
            helper.function_add(self.x, -u_vec)
            df.set_coordinates(self.mesh.geometry(), self.x)
            for bc in self.bcs:
                bc.apply(self.v.vector())
            helper.function_add(self.x, u_vec)

        # if self.bcs is not None:
            # for bc in self.bcs:
                # bc.apply(self.v.vector())

        du = h * self.v.vector().get_local()
        helper.function_add(self.x, du * 0.5)

        df.set_coordinates(self.mesh.geometry(), self.x)

        self.stress_update(h)

        helper.function_add(self.u, du)
        helper.function_add(self.x, du * 0.5)
        df.set_coordinates(self.mesh.geometry(), self.x)

        self.t += h

class CDMPlaneStrain:
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
        update_mesh = True,
    ):

        self.mesh = V.mesh()
        self.v = v0
        self.u = u0
        self.a = np.zeros_like(self.u.vector().get_local())

        self.QT = helper.quadrature_tensor_space(V, shape=(3, 3))
        self.QV = helper.quadrature_vector_space(V, dim=6)

        self.L = df.Function(self.QT)
        self.F = df.Function(self.QT) if calculate_F else None

        self.stress = df.Function(self.QV)

        self.local_q_dim = self.stress.vector().get_local().size // 6

        self.t = np.ones(self.local_q_dim) * t0

        self.ip_loop = cpp.IpLoop()
        self.ip_loop.add_law(law, np.arange(self.local_q_dim))
        self.ip_loop.resize(self.local_q_dim)

        
        self.stress_rate = stress_rate
        if stress_rate is not None:
            self.stress_rate.resize(self.local_q_dim)

        self.f_ext = f_ext
        
        test_function = df.TestFunction(V)

        self.f_int_form = df.inner(
                helper._2d_tensor_as_mandel(df.sym(df.grad(test_function))), self.stress
        ) * df.dx(
            metadata={
                "quadrature_degree": self.stress.ufl_element().degree(),
                "quadrature_scheme": "default",
            }
        )
        self.f_int =df.PETScVector()
        self.bcs = bcs
        self.bc_mesh = bc_mesh
        self.M_inv = 1 / M
        self.x = df.interpolate(df.Expression(("x[0]", "x[1]"), degree=1), V)
        self.damping_factor = damping_factor

    def __2d_tensor_to_3d(self, T):
        return df.as_matrix([[T[0,0],T[0,1],0.0],[T[1,0],T[1,1],0.0],[0.0,0.0,0.0]])

    def stress_update(self, h):

        helper.local_project(
            self.__2d_tensor_to_3d(df.nabla_grad(self.v)),
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

        self.ip_loop.update()
        helper.function_set(self.stress, new_stress)

    #@profile
    def step(self, h):

        if self.damping_factor is not None:
            c = (h * self.damping_factor) / 2
            c1 = (1 - c) / (1 + c)
            c2 = 1 / (1 + c)
        else:
            c1, c2 = 1, 1

        df.assemble(self.f_int_form, self.f_int)
        f =  self.f_ext(self.t) - self.f_int if self.f_ext is not None else - f_int
        # calculate accelerations
        self.a = self.M_inv * f.get_local()
        # given: v_n-1/2, x_n/u_n, a_n, f_int_n
        # Advance velocities and nodal positions in time
        # multiply with damping factors if needed
        helper.function_set(self.v, c1 * self.v.vector().get_local() + c2 * h * self.a)

        # if self.bcs is not None:
        if self.bc_mesh == "current":
            for bc in self.bcs:
                bc.apply(self.v.vector())
        elif self.bc_mesh == "initial":
            u_vec = helper.function_get(self.u)
            helper.function_add(self.x, -u_vec)
            df.set_coordinates(self.mesh.geometry(), self.x)
            for bc in self.bcs:
                bc.apply(self.v.vector())
            helper.function_add(self.x, u_vec)

        # if self.bcs is not None:
            # for bc in self.bcs:
                # bc.apply(self.v.vector())

        du = h * self.v.vector().get_local()
        helper.function_add(self.x, du * 0.5)

        df.set_coordinates(self.mesh.geometry(), self.x)

        self.stress_update(h)

        helper.function_add(self.u, du)
        helper.function_add(self.x, du * 0.5)
        df.set_coordinates(self.mesh.geometry(), self.x)

        self.t += h
