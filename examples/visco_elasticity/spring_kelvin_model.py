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
        self.tau = parameters["tau"] # relaxation time == eta/E1
        self.nu = parameters["nu"] # Poisson's ratio
        # lame constants (need to be updated if time dependent material parameters are used)
        self.mu0 = self.E0 / (2.0 * (1.0 + self.nu))
        self.lam0 = self.E0 * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.mu1 = self.E1 / (2.0 * (1.0 + self.nu))


        # match constraint:
        #     case Constraint.FULL:
        #         self.D = None
        #
        #     case Constraint.PLANE_STRAIN:
        #         self.D = None
        #
        #     case Constraint.PLANE_STRESS:
        #         self.D = None
        #
        #     case Constraint.UNIAXIAL_STRESS:
        #         self.D_E0 = np.array([[self.E0]])
        #     case _:
        #         msg = "Constraint not implemented"
        #         raise NotImplementedError(msg)

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
        # n_gauss = grad_del_u.size // (self.geometric_dim**2)
        mandel_view = mandel_stress.reshape(-1, self.stress_strain_dim)
        print('mandel',mandel_view)
        strain_increment = strain_from_grad_u(grad_del_u, self.constraint).reshape(-1, self.stress_strain_dim)
        print('strain inc',strain_increment)
        strain_visco_n = history['strain_visco'].reshape(-1, self.stress_strain_dim)
        print('strain visco', strain_visco_n)
        print(strain_visco_n[0], strain_visco_n[1])

        # loop over gauss points
        print('del t', del_t)
        for n, eps in enumerate(strain_increment):

            print('eps', eps)
            print('n', n)

            if del_t == 0:
                # linear step visko strain is zero
                dstress = self.E0 * eps
                D = np.array([[self.E0]])
            else:
                # compute visko strain 1D case only !!! SAME as 3D with nu=0???
                factor = (1 / del_t + 1 / self.tau + self.E0 / (self.tau * self.E1))
                deps_visko = (mandel_view[n] / (self.tau * self.E1) + self.E0 / (self.tau * self.E1) * eps -
                              strain_visco_n[n] / self.tau) / factor
                dstress = self.E0 * (eps - deps_visko)
                D = np.array([[self.E0]])* (1 - self.E0**2/(self.E1*self.tau *factor))

                print('D', D)

                # update values
                history['strain_visco'][n] += deps_visko
            mandel_view[n] += dstress
            history['strain'][n] += eps
            tangent[n] = D.flatten()


    @property
    def constraint(self) -> Constraint:
        return self._constraint

    @property
    def history_dim(self) -> None:
        return {'strain_visco': self.stress_strain_dim, 'strain': self.stress_strain_dim}

    def update(self) -> None:
        pass
