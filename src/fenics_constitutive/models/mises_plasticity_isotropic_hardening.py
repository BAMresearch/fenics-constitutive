from __future__ import annotations

import numpy as np

from fenics_constitutive import (
    IncrSmallStrainModel,
    StressStrainConstraint,
    strain_from_grad_u,
)


class VonMises3D(IncrSmallStrainModel):
    r"""
    Von Mises Plasticity model with non-linear isotropic hardening.
    Computation of trial stress state is entirely deviatoric. Volumetric part is added later
    when the stress increment for the current time step is calculated.
    
    Following are the elastic potential, plastic potential and yield surface accordingly
     
    $$
    \begin{aligned}
    & \hat{\psi}_e\left(\varepsilon_e\right) = \frac{1}{2} \kappa {e_e^2}+\mu \varepsilon{_e^{\prime}}: 
    \varepsilon{_e^{\prime}} \\
    & \hat{\psi}_p(\alpha)=\left(y_{\infty}-y_0\right)\left(-\frac{1}{\omega}+\alpha+\frac{1}{\omega} \exp (- 
    \omega \alpha)\right) \\
    & \hat{\phi}(\boldsymbol{\sigma}, \beta)=\left\|\boldsymbol{\sigma}^{\prime}\right\|-\sqrt{\frac{2} 
    {3}}\left(y_0+\beta\right) \quad \text { with } \quad \beta:=\partial_\alpha \hat{\psi}_p(\alpha)
    \end{aligned}
    $$
    
    Args:
           param: Must contain following material parameters: p_ka :  bulk modulus, p_mu : shear modulus, p_y0 : initial yield stress, p_y00 : final yield stress, p_w : saturation parameter
    """

    def __init__(self, param: dict[str, float]):
        self.xioi = np.array(
            [
                [1, 1, 1, 0, 0, 0],  # 1dyadic1 tensor
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        self.I2 = np.zeros(
            self.stress_strain_dim, dtype=np.float64
        )  # Identity of rank 2 tensor
        self.I2[0] = 1.0
        self.I2[1] = 1.0
        self.I2[2] = 1.0
        self.I4 = np.eye(
            self.stress_strain_dim, dtype=np.float64
        )  # Identity of rank 4 tensor
        self.xpp = self.I4 - (1 / 3) * self.xioi  # Projection tensor of rank 4

        self.p_ka = param["p_ka"]  # kappa - bulk modulus
        self.p_mu = param["p_mu"]  # mu - shear modulus
        self.p_y0 = param["p_y0"]  # y0 - initial yield stress
        self.p_y00 = param["p_y00"]  # y00 - final yield stress
        self.p_w = param["p_w"]  # w - saturation parameter

    def evaluate(
        self,
        time: float,
        del_t: float,
        grad_del_u: np.ndarray,
        mandel_stress: np.ndarray,
        tangent: np.ndarray,
        history: np.ndarray | dict[str, np.ndarray],
    ) -> None:
        stress_view = mandel_stress.reshape(-1, self.stress_strain_dim)
        tangent_view = tangent.reshape(-1, self.stress_strain_dim**2)
        strain_increment = strain_from_grad_u(grad_del_u, self.constraint).reshape(
            -1, self.stress_strain_dim
        )
        eps_n = history["eps_n"].reshape(-1, self.stress_strain_dim)
        alpha = history["alpha"]

        for n, eps in enumerate(strain_increment):
            tr_eps = np.sum(eps[:3])  # trace of strain
            eps_dev = eps - tr_eps * self.I2 / 3  # deviatoric strain

            # deviatoric trial internal forces (trial stress)
            del_sigtr = 2 * self.p_mu * eps_dev  # incremental trial stress
            stress_n_dev = (
                stress_view[n] - np.sum(stress_view[n][:3]) * self.I2 / 3
            )  # total deviatoric stress in last time step
            sigtr = (
                stress_n_dev + del_sigtr
            )  # total (deviatoric) trial stress in current time step

            # norm of deviatoric trial internal forces
            sigtrn = np.sqrt(np.dot(sigtr, sigtr))

            # trial yield criterion
            phitr = sigtrn - np.sqrt(2 / 3) * (
                self.p_y0
                + (self.p_y00 - self.p_y0) * (1 - np.exp(-self.p_w * alpha[n]))
            )

            # CHECK YIELD CRITERION
            # elastic - plastic step
            if phitr > 0:
                # initialization
                gamma_0 = 1
                gamma_1 = 0
                xr = 1
                it = 0
                tol = 1e-12
                tol_rel = 1e-8
                nmax = 100
                # flow direction
                xn = sigtr / sigtrn

                # UPDATE PLASTIC MULTIPLIER VIA NEWTON-RAPHSON SCHEME
                def f(x):
                    return (
                        sigtrn
                        - 2 * self.p_mu * x
                        - np.sqrt(2 / 3)
                        * (
                            self.p_y0
                            + (self.p_y00 - self.p_y0)
                            * (1 - np.exp(-self.p_w * (alpha[n] + np.sqrt(2 / 3) * x)))
                        )
                    )

                def df(x):
                    return -2 * self.p_mu - (2 / 3) * (
                        self.p_y00 - self.p_y0
                    ) * self.p_w * np.exp(-self.p_w * (alpha[n] + np.sqrt(2 / 3) * x))

                # start Newton iteration
                while np.abs(xr) > tol and abs(gamma_1 - gamma_0) > tol_rel * abs(
                    gamma_1
                ):
                    gamma_0 = gamma_1
                    it = it + 1
                    # compute residium
                    xr = f(gamma_0)
                    # compute tangent
                    xg = df(gamma_0)
                    # update plastic flow
                    gamma_1 = gamma_0 - xr / xg
                    # exit Newton algorithm for iteration > nmax
                    if it > nmax:
                        raise RuntimeError(
                            "Newton-Raphson method did not converge for plastic multiplier."
                        )
                    # end of Newton iterration

                # compute tangent with converged gamma
                xg = df(gamma_1)

                # algorithmic parameters
                xc1 = -1 / xg
                xc2 = gamma_1 / sigtrn

                # ELASTIC STEP
            else:
                xn = np.zeros(6)
                gamma_1 = 0
                xc1 = 0
                xc2 = 0

            # update eps^p_n+1 and alpha
            eps_n[n] += gamma_1 * xn
            alpha[n] += np.sqrt(2 / 3) * gamma_1

            # determine incremental elastic-plastic stresses (with volumetric part)
            sh = self.p_ka * tr_eps * self.I2 + del_sigtr - 2 * self.p_mu * gamma_1 * xn
            # update total stresses
            stress_view[n] += sh

            # determine elastic-plastic moduli
            aah = (
                self.p_ka * self.xioi
                + 2 * self.p_mu * (1 - 2 * self.p_mu * xc2) * self.xpp
                + 4 * self.p_mu * self.p_mu * (xc2 - xc1) * np.outer(xn, xn)
            )
            tangent_view[n] = aah.flatten()

    # def update(self) -> None:
    #    pass

    @property
    def constraint(self) -> StressStrainConstraint:
        return StressStrainConstraint.FULL

    @property
    def history_dim(self) -> int:
        return {"eps_n": self.constraint.stress_strain_dim, "alpha": 1}
