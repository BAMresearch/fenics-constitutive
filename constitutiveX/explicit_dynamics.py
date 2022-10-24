# import warnings

import numpy as np
import ufl

import basix
#import constitutive as c
import constitutiveX.cpp as _cpp
import dolfinx as dfx

from dolfinx_helpers import QuadratureRule, QuadratureEvaluator, get_local, set_local, add_local, set_mesh_coordinates



class CDMPlaneStrainX:
    def __init__(
        self,
        function_space,
        t0,
        f_ext,
        bcs,
        M,
        law,
        quadrature_rule,
        nonlocal_var=None,
        damping=None,
    ):
        self.mesh = function_space.mesh
        self.quadrature_rule = quadrature_rule

        self.v = dfx.fem.Function(function_space, name="Velocity")
        self.u = dfx.fem.Function(function_space, name="Displacements")

        self.f = self.v.vector.copy()
        self.damping = damping

        self.QV = self.quadrature_rule.create_quadrature_vector_space(self.mesh, 6)

        self.stress = dfx.fem.Function(self.QV)

        self.local_q_dim = get_local(self.stress).size // 6

        self.L = np.zeros(self.local_q_dim * 9)
        self.t = t0
        self.law = law

        #self.stress_rate = stress_rate

        self.f_ext = f_ext

        test_function = ufl.TestFunction(function_space)

        self.f_int_ufl = (
            -ufl.inner(
                self.__2d_tensor_as_mandel(ufl.sym(ufl.grad(test_function))),
                self.stress,
            )
            * self.quadrature_rule.dx
        )

        self.f_int_form = dfx.fem.form(self.f_int_ufl)

        self.L_evaluator = QuadratureEvaluator(
            self.__2d_tensor_to_3d(ufl.nabla_grad(self.v)),
            self.mesh,
            self.quadrature_rule,
        )
        self.bcs = bcs
        self.M = M

        self.nonlocal_var = nonlocal_var

    def __2d_tensor_to_3d(self, T):
        return ufl.as_matrix(
            [[T[0, 0], T[0, 1], 0.0], [T[1, 0], T[1, 1], 0.0], [0.0, 0.0, 0.0]]
        )

    def __2d_tensor_as_mandel(self, T):
        """
        T: 
            Symmetric 2x2 tensor
        Returns:
            Vector representation of T with factor sqrt(2) for off diagonal components
        """
        factor = 2 ** 0.5
        return ufl.as_vector([T[0, 0], T[1, 1], 0.0, 0.0, 0.0, factor * T[0, 1],])

    # @profile
    def stress_update(self, h):
        self.L_evaluator(self.L)

        input_list = [np.array([])] * _cpp.Q.LAST
        input_list[_cpp.Q.GRAD_V] = self.L
        input_list[_cpp.Q.SIGMA] = self.stress.vector.array

        _cpp.jaumann_rotate_fast_3d(self.L, self.stress.vector.array, h)
        #TODO: Is GhostUpdate really used correcxtly
        self.law.evaluate(input_list, h)
        self.stress.vector.ghostUpdate()
        # if self.nonlocal_var is not None:
        #    self.ip_loop.set(self.nonlocal_var, )
        #temp_stress = get_local(self.stress).copy()
        #if self.stress_rate is None:
        #if self.nonlocal_var is not None:
        #    self.ip_loop.set(
        #        self.nonlocal_var.Q_nonlocal,
        #        self.nonlocal_var.get_quadrature_increment(),
        #    )
        # self.ip_loop.set(c.Q.SIGMA, temp_stress)
        # self.ip_loop.set(c.Q.L, self.L)
        # self.ip_loop.set(c.Q.TIME_STEP, np.ones_like(self.t) * h)

        # self.ip_loop.evaluate()
        # temp_stress = self.ip_loop.get(c.Q.SIGMA)
        
        # self.ip_loop.update()
        # set_local(self.stress, temp_stress)

    # @profile
    def step(self, h):
        with self.f.localForm() as f_local:
            f_local.set(0.0)
        # dfx.fem.petsc.assemble_vector(self.f, dfx.fem.form(self.f_int_ufl))
        dfx.fem.petsc.assemble_vector(self.f, self.f_int_form)
        # self.f.array[:]*=-1
        if self.f_ext is not None:
            self.f.array[:] += self.f_ext(self.t).array
            # self.f.__iadd__(self.f_ext(self.t))
        self.f.ghostUpdate()

        # given: v_n-1/2, x_n/u_n, a_n, f_int_n
        # Advance velocities and nodal positions in time
        if self.damping is None:
            c1 = 1.0
            c2 = h
        else:
            c1 = (2.0 - self.damping * h) / (2.0 + self.damping * h)
            c2 = 2.0 * h / (2.0 + self.damping * h)

        set_local(self.v, c1 * get_local(self.v) + c2 * self.M.array * self.f.array)

        dfx.fem.set_bc(self.v.vector.array, self.bcs)

        # self.v.vector.ghostUpdate()

        du_half = (0.5 * h) * get_local(self.v)

        set_mesh_coordinates(self.mesh, du_half, mode="add")
        # if self.nonlocal_var is not None:
        #     self.nonlocal_var.step(
        #         h, self.law.get_internal_var(self.nonlocal_var.Q_local)
        #     )
        self.stress_update(h)

        add_local(self.u, 2.0 * du_half)

        set_mesh_coordinates(self.mesh, du_half, mode="add")

        self.t += h


class CDMNonlocalVariable:
    def __init__(
        self,
        Q_local,
        Q_nonlocal,
        t0,
        function_space,
        M,
        l,
        zeta,
        gamma,
        quadrature_rule,
    ):
        self.t = t0
        self.Q_local = Q_local
        self.Q_nonlocal = Q_nonlocal
        self.mesh = function_space.mesh
        self.M = M
        self.l = l
        self.zeta = zeta
        self.gamma = gamma
        self.quadrature_rule = quadrature_rule

        self.QS = self.quadrature_rule.create_quadrature_space(self.mesh)

        self.p_l = dfx.fem.Function(self.QS)
        self.p_nl_q = get_local(self.p_l).copy()

        self.p_nl = dfx.fem.Function(function_space)
        self.dp_nl = dfx.fem.Function(function_space)

        test_function = ufl.TestFunction(function_space)
        f_int_ufl = (
            self.l ** 2 * ufl.inner(ufl.grad(self.p_nl), ufl.grad(test_function))
            + self.p_nl * test_function
        ) * self.quadrature_rule.dx
        f_ext_ufl = self.p_l * test_function * self.quadrature_rule.dx

        self.f_ufl = -f_int_ufl + f_ext_ufl

        self.f_form = dfx.fem.form(self.f_ufl)
        self.f = self.p_nl.vector.copy()
        self.delta_t = dfx.fem.Constant(self.mesh, 0.0)
        # self.p_evaluator = QuadratureEvaluator(
            # self.delta_t * self.dp_nl, self.mesh, self.quadrature_rule
        # )
        self.p_evaluator = QuadratureEvaluator(
            self.p_nl, self.mesh, self.quadrature_rule
        )

    def step(self, h, p):
        with self.f.localForm() as f_local:
            f_local.set(0.0)

        self.f.ghostUpdate()
        self.delta_t.value = h

        set_local(self.p_l, p)

        dfx.fem.petsc.assemble_vector(self.f, self.f_form)

        self.f.ghostUpdate()

        c = self.gamma / self.zeta
        c1 = (2.0 - c * h) / (2.0 + c * h)
        c2 = 2.0 * h / (2.0 + c * h)

        set_local(
            self.dp_nl, c1 * get_local(self.dp_nl) + c2 * self.M.array * self.f.array
        )
        # set_local(
            # self.dp_nl, get_local(self.dp_nl) + h * self.M.array * self.f.array
        # )
        # if np.sum(self.p_nl_q < 0) > 0:
            # print("Warning: negative increment in nonlocal strain")

        # del_p_nl = h * get_local(self.dp_nl)
        add_local(self.p_nl, h * get_local(self.dp_nl))

        self.p_evaluator(self.p_nl_q)
        self.t += h

    def get_quadrature_increment(self):
        return self.p_nl_q

    def get_nodal_values(self):
        return self.p_nl

class CDM3D:
    def __init__(
        self,
        function_space,
        t0,
        f_ext,
        bcs,
        M,
        law,
        quadrature_rule,
        stress_rate=None,
        nonlocal_var=None,
        damping=None,
        project=False,
    ):
        self.mesh = function_space.mesh
        self.quadrature_rule = quadrature_rule

        self.v = dfx.fem.Function(function_space)
        self.u = dfx.fem.Function(function_space)
        # self.a = dfx.fem.Function(function_space)

        self.f = self.v.vector.copy()
        self.project = project
        self.damping = damping

        # project only for debugging purposes, very slow
        if self.project:
            self.QT = self.quadrature_rule.create_quadrature_tensor_space(
                self.mesh, (3, 3)
            )
            self.L_fun = dfx.fem.Function(self.QT)
        self.QV = self.quadrature_rule.create_quadrature_vector_space(self.mesh, 6)

        self.stress = dfx.fem.Function(self.QV)

        self.local_q_dim = get_local(self.stress).size // 6

        self.L = np.zeros(self.local_q_dim * 9)
        self.t = np.ones(self.local_q_dim) * t0
        self.law = law
        self.ip_loop = c.IpLoop()
        self.ip_loop.add_law(law, np.arange(self.local_q_dim))
        self.ip_loop.resize(self.local_q_dim)

        self.stress_rate = stress_rate
        if stress_rate is not None:
            self.stress_rate.resize(self.local_q_dim)

        self.f_ext = f_ext

        test_function = ufl.TestFunction(function_space)

        self.f_int_ufl = (
            -ufl.inner(
                self.as_mandel(ufl.sym(ufl.grad(test_function))),
                self.stress,
            )
            * self.quadrature_rule.dx
        )

        self.f_int_form = dfx.fem.form(self.f_int_ufl)

        self.L_evaluator = QuadratureEvaluator(
            ufl.nabla_grad(self.v),
            self.mesh,
            self.quadrature_rule,
        )
        self.bcs = bcs
        self.M = M

        self.nonlocal_var = nonlocal_var


    def as_mandel(self, T):
        """
        T: 
            Symmetric 3x3 tensor
        Returns:
            Vector representation of T with factor sqrt(2) for off diagonal components
        """
        factor = 2 ** 0.5
        return ufl.as_vector([T[0, 0], T[1, 1], T[2,2], factor * T[2,1], factor * T[0,2], factor * T[0, 1],])

    # @profile
    def stress_update(self, h):
        if self.project:
            project(
                ufl.nabla_grad(self.v),
                self.QT,
                self.quadrature_rule.dx,
                u=self.L_fun,
            )
            self.L = get_local(self.L_fun)
        else:
            self.L_evaluator(self.L)
        # if self.nonlocal_var is not None:
        #    self.ip_loop.set(self.nonlocal_var, )
        temp_stress = get_local(self.stress).copy()
        if self.stress_rate is None:
            if self.nonlocal_var is not None:
                self.ip_loop.set(
                    self.nonlocal_var.Q_nonlocal,
                    self.nonlocal_var.get_quadrature_increment(),
                )
            self.ip_loop.set(c.Q.SIGMA, temp_stress)
            self.ip_loop.set(c.Q.L, self.L)
            self.ip_loop.set(c.Q.TIME_STEP, np.ones_like(self.t) * h)

            self.ip_loop.evaluate()
            temp_stress = self.ip_loop.get(c.Q.SIGMA)
        else:
            if self.nonlocal_var is not None:
                self.ip_loop.set(
                    self.nonlocal_var.Q_nonlocal,
                    self.nonlocal_var.get_quadrature_values(),
                )
            self.stress_rate.set(self.L)
            temp_stress = self.stress_rate(temp_stress, h * 0.5)

            self.ip_loop.set(c.Q.SIGMA, temp_stress)
            self.ip_loop.set(c.Q.EPS, c.strain_increment(self.L, h))

            self.ip_loop.evaluate()

            temp_stress = self.ip_loop.get(c.Q.SIGMA)
            temp_stress = self.stress_rate(temp_stress, h * 0.5)

        self.ip_loop.update()
        set_local(self.stress, temp_stress)

    # @profile
    def step(self, h):
        with self.f.localForm() as f_local:
            f_local.set(0.0)
        # dfx.fem.petsc.assemble_vector(self.f, dfx.fem.form(self.f_int_ufl))
        dfx.fem.petsc.assemble_vector(self.f, self.f_int_form)
        # self.f.array[:]*=-1
        if self.f_ext is not None:
            self.f.array[:] += self.f_ext(self.t).array
            # self.f.__iadd__(self.f_ext(self.t))
        self.f.ghostUpdate()

        # given: v_n-1/2, x_n/u_n, a_n, f_int_n
        # Advance velocities and nodal positions in time
        if self.damping is None:
            c1 = 1.0
            c2 = h
        else:
            c1 = (2.0 - self.damping * h) / (2.0 + self.damping * h)
            c2 = 2.0 * h / (2.0 + self.damping * h)

        set_local(self.v, c1 * get_local(self.v) + c2 * self.M.array * self.f.array)

        dfx.fem.set_bc(self.v.vector.array, self.bcs)

        # self.v.vector.ghostUpdate()

        du_half = (0.5 * h) * get_local(self.v)

        set_mesh_coordinates(self.mesh, du_half, mode="add")
        if self.nonlocal_var is not None:
            self.nonlocal_var.step(
                h, self.law.get_internal_var(self.nonlocal_var.Q_local)
            )
        self.stress_update(h)

        add_local(self.u, 2.0 * du_half)

        set_mesh_coordinates(self.mesh, du_half, mode="add")

        self.t += h
