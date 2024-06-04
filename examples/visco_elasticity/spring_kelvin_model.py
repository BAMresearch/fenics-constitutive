from __future__ import annotations

import numpy as np

from fenics_constitutive import Constraint, IncrSmallStrainModel, strain_from_grad_u


class SpringKelvinModel(IncrSmallStrainModel):
    ''' viscoelastic model based on 1D Three Parameter Model with spring and Kelvin body in row

                               |--- E_1: spring ---|
           --- E_0: spring  ---|                   |--
                               |--- eta: damper ---|

    with deviatoric assumptions for 3D generalization (volumetric part of visco strain == 0 damper just working on deviatoric part)
    time integration: backward Euler

    '''
    def __init__(self, parameters: dict[str, float], constraint: Constraint):
        self._constraint = constraint
        self.E0 = parameters["E0"] # elastic modulus
        self.E1 = parameters["E1"] # visco modulus
        self.tau = parameters["tau"] # relaxation time == eta/(2 mu1) for 1D case eta/E1
        if constraint == Constraint.UNIAXIAL_STRESS:
            self.nu = 0.0
        else:
            self.nu = parameters["nu"] # Poisson's ratio

        # lame constants (need to be updated if time dependent material parameters are used)
        self.mu0 = self.E0 / (2.0 * (1.0 + self.nu))
        self.lam0 = self.E0 * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.mu1 = self.E1 / (2.0 * (1.0 + self.nu))

        self.I2 = np.zeros(self.stress_strain_dim, dtype=np.float64)  # Identity of rank 2 tensor
        self.compute_elasticity() # initialize elasticity tensor

    def compute_elasticity(self):
        match self._constraint:
            case Constraint.FULL:
                self.D_0 = np.array(
                    [
                        [2.0 * self.mu0 + self.lam0, self.lam0, self.lam0, 0.0, 0.0, 0.0],
                        [self.lam0, 2.0 * self.mu0 + self.lam0, self.lam0, 0.0, 0.0, 0.0],
                        [self.lam0, self.lam0, 2.0 * self.mu0 + self.lam0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 2.0 * self.mu0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 2.0 * self.mu0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 2.0 * self.mu0],
                    ]
                )
                self.I2[0] = 1.0
                self.I2[1] = 1.0
                self.I2[2] = 1.0

            case Constraint.PLANE_STRAIN:
                self.D_0 = np.array(
                    [
                        [2.0 * self.mu0 + self.lam0, self.lam0, self.lam0, 0.0],
                        [self.lam0, 2.0 * self.mu0 + self.lam0, self.lam0, 0.0],
                        [self.lam0, self.lam0, 2.0 * self.mu0 + self.lam0, 0.0],
                        [0.0, 0.0, 0.0, 2.0 * self.mu0],
                    ]
                )
                self.I2[0] = 1.0
                self.I2[1] = 1.0
                self.I2[2] = 1.0

            case Constraint.PLANE_STRESS:
                self.D_0 = (
                        self.E0
                        / (1 - self.nu ** 2.0)
                        * np.array(
                    [
                        [1.0, self.nu, 0.0, 0.0],
                        [self.nu, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, (1.0 - self.nu)],
                    ]
                )
                )
                self.I2[0] = 1.0
                self.I2[1] = 1.0

            case Constraint.UNIAXIAL_STRESS:
                self.D_0 = np.array([[self.E0]])
                self.I2[0] = 1.0
            case _:
                msg = "Constraint not implemented"
                raise NotImplementedError(msg)

    def evaluate(
        self,
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
        n_gauss = grad_del_u.size // (self.geometric_dim ** 2)
        mandel_view = mandel_stress.reshape(-1, self.stress_strain_dim)

        strain_increment = strain_from_grad_u(grad_del_u, self.constraint).reshape(-1, self.stress_strain_dim)
        strain_visco_n = history['strain_visco'].reshape(-1, self.stress_strain_dim)
        strain_n = history['strain'].reshape(-1, self.stress_strain_dim)

        I2 = np.tile(self.I2, n_gauss).reshape(-1, self.stress_strain_dim)
        tr_eps = np.sum(strain_increment[:, :self.geometric_dim], axis=1)[:, np.newaxis]

        if del_t == 0:
            # linear step visko strain is zero
            D = self.D_0
            mandel_view += strain_increment @ D
            _deps_visko = np.zeros_like(strain_increment)
        else:
            # visco step
            factor = (1 / del_t + 1 / self.tau + self.mu0 / (self.tau * self.mu1))

            _deps_visko = 1 / factor * (
                        1 / (self.tau * 2 * self.mu1) * mandel_view
                        - 1 / self.tau * strain_visco_n
                        + self.mu0 / (self.tau * self.mu1) * strain_increment
                        + self.lam0 / (self.tau * 2 * self.mu1) * tr_eps * I2
                )

            mandel_view += strain_increment @ self.D_0 - 2 * self.mu0 * _deps_visko
            D = (1 - self.mu0 / (self.tau * self.mu1 * factor)) * self.D_0

        tangent[:] = np.tile(D.flatten(), n_gauss)
        strain_visco_n += _deps_visko
        strain_n += strain_increment


    @property
    def constraint(self) -> Constraint:
        return self._constraint

    @property
    def history_dim(self) -> None:
        return {'strain_visco': self.stress_strain_dim, 'strain': self.stress_strain_dim}

    def update(self) -> None:
        pass

