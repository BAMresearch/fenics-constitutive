from __future__ import annotations

import numpy as np

from fenics_constitutive import (
    IncrSmallStrainModel,
    StressStrainConstraint,
    strain_from_grad_u,
)


class LinearElasticityModel(IncrSmallStrainModel):
    """
    A linear elastic material model which has been implemented for all constraints.

    Args:
        parameters: Material parameters. Must contain "E" for the Youngs modulus and "nu" for the Poisson ratio.
        constraint: Constraint type.
    """

    def __init__(
        self, parameters: dict[str, float], constraint: StressStrainConstraint
    ):
        self._constraint = constraint
        E = parameters["E"]
        nu = parameters["nu"]
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        match constraint:
            case StressStrainConstraint.FULL:
                # see https://en.wikipedia.org/wiki/Hooke%27s_law
                self.D = np.array(
                    [
                        [2.0 * mu + lam, lam, lam, 0.0, 0.0, 0.0],
                        [lam, 2.0 * mu + lam, lam, 0.0, 0.0, 0.0],
                        [lam, lam, 2.0 * mu + lam, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 2.0 * mu, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 2.0 * mu, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 2.0 * mu],
                    ]
                )
            case StressStrainConstraint.PLANE_STRAIN:
                # We assert that the strain is being provided with 0 in the z-direction
                # see https://en.wikipedia.org/wiki/Hooke%27s_law
                self.D = np.array(
                    [
                        [2.0 * mu + lam, lam, lam, 0.0],
                        [lam, 2.0 * mu + lam, lam, 0.0],
                        [lam, lam, 2.0 * mu + lam, 0.0],
                        [0.0, 0.0, 0.0, 2.0 * mu],
                    ]
                )
            case StressStrainConstraint.PLANE_STRESS:
                # We do not make any assumptions about strain in the z-direction
                # This matrix just multiplies the z component by 0.0 which results
                # in a plane stress state
                # see https://en.wikipedia.org/wiki/Hooke%27s_law
                self.D = (
                    E
                    / (1 - nu**2.0)
                    * np.array(
                        [
                            [1.0, nu, 0.0, 0.0],
                            [nu, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, (1.0 - nu)],
                        ]
                    )
                )
            case StressStrainConstraint.UNIAXIAL_STRAIN:
                # see https://csmbrannon.net/2012/08/02/distinction-between-uniaxial-stress-and-uniaxial-strain/
                C = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
                self.D = np.array([[C]])
            case StressStrainConstraint.UNIAXIAL_STRESS:
                # see https://csmbrannon.net/2012/08/02/distinction-between-uniaxial-stress-and-uniaxial-strain/
                self.D = np.array([[E]])
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
        n_gauss = grad_del_u.size // (self.geometric_dim**2)
        mandel_view = mandel_stress.reshape(-1, self.stress_strain_dim)
        strain_increment = strain_from_grad_u(grad_del_u, self.constraint)
        mandel_view += strain_increment.reshape(-1, self.stress_strain_dim) @ self.D
        tangent[:] = np.tile(self.D.flatten(), n_gauss)

    @property
    def constraint(self) -> StressStrainConstraint:
        return self._constraint

    @property
    def history_dim(self) -> None:
        return None

    def update(self) -> None:
        pass
