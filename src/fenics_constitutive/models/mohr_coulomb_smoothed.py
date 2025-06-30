import numpy as np
from fenics_constitutive import Constraint, IncrSmallStrainModel, strain_from_grad_u
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

class mohr_coulomb_smoothed_3D(IncrSmallStrainModel):

    """Von Mises Plasticity model with linear isotropic hardening.
    Computation of trial stress state is entirely deviatoric. Volumetric part is added later
    when the stress increment for the current time step is calculated """

    def __init__(self, param: dict[str, float]):

        self.xioi = np.array([[1, 1, 1, 0, 0, 0], #1dyadic1 tensor
                         [1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0]])
        self.tr = np.zeros(self.stress_strain_dim, dtype=np.float64)  # Identity of rank 2 tensor
        self.tr[0] = 1.0
        self.tr[1] = 1.0
        self.tr[2] = 1.0
        self.I4 = np.eye(self.stress_strain_dim, dtype=np.float64)  # Identity of rank 4 tensor
        self.dev = self.I4 - (1 / 3) * self.xioi # Projection tensor of rank 4

        self.E = param["E"]  # Young's modulus
        self.nu = param["nu"]  # Poisson ratio
        self.c = param["c"]   # cohesion
        self.phi = param["phi"]   # friction angle
        self.psi = param["psi"]   # dilatancy angle
        self.theta_T = param["theta_T"]   # transition angle as defined by Abbo and Sloan
        self.a = param["a"]   # tension cuff-off parameter

        self.lmbda = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.mu = self.E / (2.0 * (1.0 + self.nu))

        self.C_elas = np.array([
            [self.lmbda + 2 * self.mu, self.lmbda, self.lmbda, 0, 0, 0],
            [self.lmbda, self.lmbda + 2 * self.mu, self.lmbda, 0, 0, 0],
            [self.lmbda, self.lmbda, self.lmbda + 2 * self.mu, 0, 0, 0],
            [0, 0, 0, 2 * self.mu, 0, 0],
            [0, 0, 0, 0, 2 * self.mu, 0],
            [0, 0, 0, 0, 0, 2 * self.mu],
        ])

        self.ZERO_VECTOR = np.zeros(self.stress_strain_dim)

    def evaluate(
        self,
        del_t: float,
        grad_del_u: np.ndarray,
        mandel_stress: np.ndarray,
        tangent: np.ndarray,
        history: np.ndarray | dict[str, np.ndarray],
    ) -> None:

        stress_view = mandel_stress.reshape(-1, self.stress_strain_dim)
        tangent_view = tangent.reshape(-1, self.stress_strain_dim**2)
        strain_increment = strain_from_grad_u(grad_del_u, self.constraint).reshape(-1, self.stress_strain_dim)
        # eps_n = history["eps_n"].reshape(-1, self.stress_strain_dim)
        lmda = history["lmda"]

        # if type(self.p_mu) is np.ndarray:
        #     time = True
        # else:
        #     p_mu = self.p_mu

        def mandel_to_tensor(mandel):
            """Convert Mandel stress vector (6x1) to symmetric 3x3 tensor."""
            return jnp.array([
                [mandel[0], mandel[3] / jnp.sqrt(2), mandel[5] / jnp.sqrt(2)],
                [mandel[3] / jnp.sqrt(2), mandel[1], mandel[4] / jnp.sqrt(2)],
                [mandel[5] / jnp.sqrt(2), mandel[4] / jnp.sqrt(2), mandel[2]]
            ])

        def J3(s):
            s_tensor = mandel_to_tensor(s)
            return jnp.linalg.det(s_tensor)

        def J2(s):
            return 0.5 * jnp.vdot(s, s)

        def theta(s):
            # J2_ = J2(s)
            epsilon = 1e-8
            J2_ = J2(s) + epsilon  # Regularization
            # jax.debug.print("J2_ is {}:",J2_)
            arg = -(3.0 * np.sqrt(3.0) * J3(s)) / (2.0 * jnp.sqrt(J2_ * J2_ * J2_))
            arg = jnp.clip(arg, -1.0, 1.0)
            theta = 1.0 / 3.0 * jnp.arcsin(arg)
            return theta

        def sign(x):
            return jax.lax.cond(x < 0.0, lambda x: -1, lambda x: 1, x)

        def coeff1(theta, angle):
            return np.cos(self.theta_T) - (1.0 / np.sqrt(3.0)) * np.sin(angle) * np.sin(self.theta_T)

        def coeff2(theta, angle):
            return sign(theta) * np.sin(self.theta_T) + (1.0 / np.sqrt(3.0)) * np.sin(angle) * np.cos(self.theta_T)

        coeff3 = 18.0 * np.cos(3.0 * self.theta_T) * np.cos(3.0 * self.theta_T) * np.cos(3.0 * self.theta_T)

        def C(theta, angle):
            return (
                    -np.cos(3.0 * self.theta_T) * coeff1(theta, angle) - 3.0 * sign(theta) * np.sin(3.0 * self.theta_T) * coeff2(
                theta, angle)
            ) / coeff3

        def B(theta, angle):
            return (
                    sign(theta) * np.sin(6.0 * self.theta_T) * coeff1(theta, angle) - 6.0 * np.cos(6.0 * self.theta_T) * coeff2(
                theta, angle)
            ) / coeff3

        def A(theta, angle):
            return (
                    -(1.0 / np.sqrt(3.0)) * np.sin(angle) * sign(theta) * np.sin(self.theta_T)
                    - B(theta, angle) * sign(theta) * np.sin(3 * self.theta_T)
                    - C(theta, angle) * np.sin(3.0 * self.theta_T) * np.sin(3.0 * self.theta_T)
                    + np.cos(self.theta_T)
            )

        def K(theta, angle):
            def K_false(theta):
                return jnp.cos(theta) - (1.0 / np.sqrt(3.0)) * np.sin(angle) * jnp.sin(theta)

            def K_true(theta):
                return (
                        A(theta, angle)
                        + B(theta, angle) * jnp.sin(3.0 * theta)
                        + C(theta, angle) * jnp.sin(3.0 * theta) * jnp.sin(3.0 * theta)
                )

            return jax.lax.cond(jnp.abs(theta) > self.theta_T, K_true, K_false, theta)

        def a_g(angle):
            return self.a * np.tan(self.phi) / np.tan(angle)

        def surface(sigma_local, angle):
            s = self.dev @ sigma_local

            I1 = self.tr @ sigma_local
            # jax.debug.print("I1 is {}:",I1)
            theta_ = theta(s)
            return (
                    (I1 / 3.0 * np.sin(angle))
                    + jnp.sqrt(
                J2(s) * K(theta_, angle) * K(theta_, angle) + a_g(angle) * a_g(angle) * np.sin(angle) * np.sin(angle)
            )
                    - self.c * np.cos(angle)
            )

        def f(sigma_local):
            return surface(sigma_local, self.phi)

        def g(sigma_local):
            return surface(sigma_local, self.psi)

        dgdsigma = jax.jacfwd(g) # this derivative needs to be replaced by analytical one

        def deps_p(sigma_local, dlambda, deps_local, sigma_n_local):
            sigma_elas_local = sigma_n_local + self.C_elas @ deps_local

            yielding = f(sigma_elas_local)

            def deps_p_elastic(sigma_local, dlambda):
                return self.ZERO_VECTOR

            def deps_p_plastic(sigma_local, dlambda):
                # jax.debug.print("sigma_elas_local: {}", sigma_elas_local)
                # jax.debug.print("dgdsigma: {}", dgdsigma(sigma_local))
                return dlambda * dgdsigma(sigma_local)

            return jax.lax.cond(yielding <= 0.0, deps_p_elastic, deps_p_plastic, sigma_local, dlambda)

        def r_g(sigma_local, dlambda, deps_local, sigma_n_local):
            deps_p_local = deps_p(sigma_local, dlambda, deps_local, sigma_n_local)
            return sigma_local - sigma_n_local - self.C_elas @ (deps_local - deps_p_local)

        def r_f(sigma_local, dlambda, deps_local, sigma_n_local):
            sigma_elas_local = sigma_n_local + self.C_elas @ deps_local
            yielding = f(sigma_elas_local)

            def r_f_elastic(sigma_local, dlambda):
                return dlambda

            def r_f_plastic(sigma_local, dlambda):
                return f(sigma_local)

            return jax.lax.cond(yielding <= 0.0, r_f_elastic, r_f_plastic, sigma_local, dlambda)

        def r(y_local, deps_local, sigma_n_local):
            sigma_local = y_local[:self.stress_strain_dim]
            dlambda_local = y_local[-1]

            res_g = r_g(sigma_local, dlambda_local, deps_local, sigma_n_local)
            res_f = r_f(sigma_local, dlambda_local, deps_local, sigma_n_local)

            res = jnp.c_["0,1,-1", res_g, res_f]  # concatenates an array and a scalar
            return res

        drdy = jax.jacfwd(r) # this derivative needs to be replaced by analytical one

        Nitermax, tol = 200, 1e-10

        ZERO_SCALAR = np.array([0.0])

        def return_mapping(deps_local, sigma_n_local):
            """Performs the return-mapping procedure.

            It solves elastoplastic constitutive equations numerically by applying the
            Newton method in a single Gauss point. The Newton loop is implement via
            `jax.lax.while_loop`.

            The function returns `sigma_local` two times to reuse its values after
            differentiation, i.e. as once we apply
            `jax.jacfwd(return_mapping, has_aux=True)` the ouput function will
            have an output of
            `(C_tang_local, (sigma_local, niter_total, yielding, norm_res, dlambda))`.

            Returns:
                sigma_local: The stress at the current Gauss point.
                niter_total: The total number of iterations.
                yielding: The value of the yield function.
                norm_res: The norm of the residuals.
                dlambda: The value of the plastic multiplier.
            """
            niter = 0
            # dlambda = jnp.array([dlambda_n])
            dlambda = ZERO_SCALAR
            # print("lambda is: ", dlambda)
            sigma_local = sigma_n_local
            y_local = jnp.concatenate([sigma_local, dlambda])

            res = r(y_local, deps_local, sigma_n_local)
            norm_res0 = jnp.linalg.norm(res)

            def cond_fun(state):
                norm_res, niter, _ = state
                return jnp.logical_and(norm_res / norm_res0 > tol, niter < Nitermax)

            def body_fun(state):
                norm_res, niter, history = state

                y_local, deps_local, sigma_n_local, res = history

                j = drdy(y_local, deps_local, sigma_n_local)
                # j = finite_difference_jacobian_drdy(y_local, deps_local, sigma_n_local)
                # jax.debug.print("j is {}:",j)
                j_inv_vp = jnp.linalg.solve(j, -res)
                y_local = y_local + j_inv_vp

                res = r(y_local, deps_local, sigma_n_local)
                norm_res = jnp.linalg.norm(res)
                history = y_local, deps_local, sigma_n_local, res

                niter += 1

                return (norm_res, niter, history)

            history = (y_local, deps_local, sigma_n_local, res)

            norm_res, niter_total, y_local = jax.lax.while_loop(cond_fun, body_fun, (norm_res0, niter, history))

            sigma_local = y_local[0][:self.stress_strain_dim]
            # jax.debug.print("sigma_local is {}:", sigma_local)
            # jax.debug.print("dgdsigma is {}:", dgdsigma(sigma_local))
            dlambda = y_local[0][-1]
            sigma_elas_local = self.C_elas @ deps_local
            yielding = f(sigma_n_local + sigma_elas_local)

            return sigma_local, (sigma_local, niter_total, yielding, norm_res, dlambda)

        dsigma_ddeps = jax.jacfwd(return_mapping, has_aux=True)

        dsigma_ddeps_vec = jax.jit(jax.vmap(dsigma_ddeps, in_axes=(0, 0)))

        (C_tang_global, state) = dsigma_ddeps_vec(strain_increment, stress_view)
        sigma_global, niter, yielding, norm_res, dlambda = state

        # print(sigma_global)



        for n, eps in enumerate(strain_increment):

            stress_view[n] = sigma_global[n,:].reshape(-1)

            if np.all(eps == 0):
                tangent_view[n] = self.C_elas.flatten()
            else:
                tangent_view[n] = C_tang_global[n,:,:].flatten()

        # for n, eps in enumerate(strain_increment):
        #
        #     # stress_new, (_, niter, yielding, norm_res, dlambda_new) = return_mapping(eps, stress_view[n])
        #     # lmda[n] += dlambda_new
        #     # stress_view[n] = stress_new
        #     # dsigma_ddeps = jax.jacfwd(return_mapping, has_aux=True)
        #     # tangent_view[n] = dsigma_ddeps()
        #
        #     (C_tang_global, state) = dsigma_ddeps(eps, stress_view[n])
        #     # print("eps", eps)
        #     # print("C_tang_global * eps", C_tang_global @ eps)
        #     sigma_global, niter, yielding, norm_res, dlambda = state
        #     # print("sigma_global", sigma_global)
        #     print("yielding", yielding,
        #           "norm_res", norm_res)
        #     print("stress : ", sigma_global)
        #
        #     stress_view[n] = sigma_global
        #     lmda[n] += dlambda
        #     if np.all(eps == 0):
        #         tangent_view[n] = self.C_elas.flatten()
        #     else:
        #         tangent_view[n] = C_tang_global.flatten()

    def update(self) -> None:
        pass

    @property
    def constraint(self) -> Constraint:
        return Constraint.FULL

    @property
    def history_dim(self) -> int:
        return {"lmda": 1}