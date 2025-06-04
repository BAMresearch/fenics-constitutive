from __future__ import annotations

import numpy as np

from fenics_constitutive import (
    IncrSmallStrainModel,
    StressStrainConstraint,
    strain_from_grad_u,
)


class SpringMaxwellModel(IncrSmallStrainModel):
    """viscoelastic model based on 1D Three Parameter Model with spring and Maxwell body in parallel

             |----------- E_0: spring  ----------|
           --|                                   |--
             |--- E_1: spring --- eta: damper ---|

    with deviatoric assumptions for 3D generalization (volumetric part of visco strain == 0 damper just working on deviatoric part)
    time integration: backward Euler

    Args:
        parameters: Material parameters. Must contain "E0" for the elastic Youngs modulus, "E1" for the viscous modulus and "tau" for the relaxation time.
        constraint: Constraint type.

    """

    def __init__(
        self, parameters: dict[str, float], constraint: StressStrainConstraint
    ):
        self._constraint = constraint
        self.E0 = parameters["E0"]  # elastic modulus
        self.E1 = parameters["E1"]  # visco modulus
        self.tau = parameters[
            "tau"
        ]  # relaxation time == eta/(2 mu1) for 1D case eta/E1
        if constraint == StressStrainConstraint.UNIAXIAL_STRESS:
            self.nu = 0.0
        else:
            self.nu = parameters["nu"]  # Poisson's ratio

        # lame constants (need to be updated if time dependent material parameters are used)
        self.mu0 = self.E0 / (2.0 * (1.0 + self.nu))
        self.lam0 = self.E0 * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.mu1 = self.E1 / (2.0 * (1.0 + self.nu))
        self.lam1 = self.E1 * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

        self.compute_elasticity()  # initialize elasticity tensor

    def compute_elasticity(self):
        match self._constraint:
            case StressStrainConstraint.FULL:
                self.D_0 = np.array(
                    [
                        [
                            2.0 * self.mu0 + self.lam0,
                            self.lam0,
                            self.lam0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        [
                            self.lam0,
                            2.0 * self.mu0 + self.lam0,
                            self.lam0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        [
                            self.lam0,
                            self.lam0,
                            2.0 * self.mu0 + self.lam0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        [0.0, 0.0, 0.0, 2.0 * self.mu0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 2.0 * self.mu0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 2.0 * self.mu0],
                    ]
                )
                self.D_1 = np.array(
                    [
                        [
                            2.0 * self.mu1 + self.lam1,
                            self.lam1,
                            self.lam1,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        [
                            self.lam1,
                            2.0 * self.mu1 + self.lam1,
                            self.lam1,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        [
                            self.lam1,
                            self.lam1,
                            2.0 * self.mu1 + self.lam1,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        [0.0, 0.0, 0.0, 2.0 * self.mu1, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 2.0 * self.mu1, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 2.0 * self.mu1],
                    ]
                )

            case StressStrainConstraint.PLANE_STRAIN:
                self.D_0 = np.array(
                    [
                        [2.0 * self.mu0 + self.lam0, self.lam0, self.lam0, 0.0],
                        [self.lam0, 2.0 * self.mu0 + self.lam0, self.lam0, 0.0],
                        [self.lam0, self.lam0, 2.0 * self.mu0 + self.lam0, 0.0],
                        [0.0, 0.0, 0.0, 2.0 * self.mu0],
                    ]
                )
                self.D_1 = np.array(
                    [
                        [2.0 * self.mu1 + self.lam1, self.lam1, self.lam1, 0.0],
                        [self.lam1, 2.0 * self.mu1 + self.lam1, self.lam1, 0.0],
                        [self.lam1, self.lam1, 2.0 * self.mu1 + self.lam1, 0.0],
                        [0.0, 0.0, 0.0, 2.0 * self.mu1],
                    ]
                )

            case StressStrainConstraint.PLANE_STRESS:
                self.D_0 = (
                    self.E0
                    / (1 - self.nu**2.0)
                    * np.array(
                        [
                            [1.0, self.nu, 0.0, 0.0],
                            [self.nu, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, (1.0 - self.nu)],
                        ]
                    )
                )
                self.D_1 = (
                    self.E1
                    / (1 - self.nu**2.0)
                    * np.array(
                        [
                            [1.0, self.nu, 0.0, 0.0],
                            [self.nu, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, (1.0 - self.nu)],
                        ]
                    )
                )

            case StressStrainConstraint.UNIAXIAL_STRESS:
                self.D_0 = np.array([[self.E0]])
                self.D_1 = np.array([[self.E1]])
            case _:
                msg = "Constraint not implemented"
                raise NotImplementedError(msg)

    def evaluate(
        self,
        time: float,
        del_t: float,
        grad_del_u: np.ndarray,
        mandel_stress: np.ndarray,
        tangent: np.ndarray,
        history: np.ndarray | dict[str, np.ndarray] | None,
    ) -> None:
        assert (
            grad_del_u.size // (self.geometric_dim**2)
            == mandel_stress.size // self.stress_strain_dim
            == tangent.size // (self.stress_strain_dim**2)
        )

        # reshape gauss point arrays
        n_gauss = grad_del_u.size // (self.geometric_dim**2)
        mandel_view = mandel_stress.reshape(-1, self.stress_strain_dim)

        strain_increment = strain_from_grad_u(grad_del_u, self.constraint).reshape(
            -1, self.stress_strain_dim
        )
        strain_visco_n = history["strain_visco"].reshape(-1, self.stress_strain_dim)
        strain_n = history["strain"].reshape(-1, self.stress_strain_dim)

        # if del_t == 0:
        #     # linear step visko strain is zero
        #     D = self.D_0 + self.D_1
        #     mandel_view += strain_increment @ D
        #     _deps_visko = np.zeros_like(strain_increment)
        # else:
        assert del_t > 0, "Time step must be defined and positive."

        strain_total = strain_n + strain_increment
        factor = 1 / del_t + 1 / self.tau
        _deps_visko = (
            1
            / factor
            * (
                1 / (self.tau * 2 * self.mu1) * strain_total @ self.D_1
                - 1 / self.tau * strain_visco_n
            )
        )

        dstress = strain_increment @ (self.D_0 + self.D_1) - 2 * self.mu1 * _deps_visko
        mandel_view += dstress
        D = self.D_0 + (1 - 1 / (self.tau * factor)) * self.D_1

        tangent[:] = np.tile(D.flatten(), n_gauss)
        strain_visco_n += _deps_visko
        strain_n += strain_increment

    @property
    def constraint(self) -> StressStrainConstraint:
        return self._constraint

    @property
    def history_dim(self) -> None:
        return {
            "strain_visco": self.stress_strain_dim,
            "strain": self.stress_strain_dim,
        }

    # def update(self) -> None:
    #    pass
