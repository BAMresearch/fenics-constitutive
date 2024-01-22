from fenics_constitutive import Constraint, IncrSmallStrainModel, strain_from_grad_u
import numpy as np


class LinearElasticityModel(IncrSmallStrainModel):
    def __init__(self, E: float, nu: float, constraint: Constraint):
        self._constraint = constraint
        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        match constraint:
            case Constraint.FULL:
                self.D = np.array(
                    [
                        [2.0 * mu + lam, lam, lam, 0.0, 0.0, 0.0],
                        [lam, 2.0 * mu + lam, lam, 0.0, 0.0, 0.0],
                        [lam, lam, 2.0 * mu + lam, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, mu, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, mu, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, mu],
                    ]
                )
            case _:
                raise NotImplementedError("Only full constraint implemented")

    def evaluate(
        self,
        del_t: float,
        grad_del_u: np.ndarray,
        mandel_stress: np.ndarray,
        tangent: np.ndarray,
        history: np.ndarray | dict[str, np.ndarray] | None,
    ) -> None:
        assert (
            grad_del_u.size / (self.geometric_dim**2)
            == mandel_stress.size / self.stress_strain_dim
            == tangent.size / (self.stress_strain_dim**2)
        )
        n_gauss = grad_del_u.shape / (self.geometric_dim**2)

        strain_increment = strain_from_grad_u(grad_del_u, self.constraint)
        mandel_stress += strain_increment.reshape(-1, self.stress_strain_dim) @ self.D
        tangent[:] = np.tile(self.D, n_gauss)
    
    @property
    def constraint(self) -> Constraint:
        return self._constraint
    
    @property
    def history_dim(self) -> None:
        return None
    
    def update(self) -> None:
        pass