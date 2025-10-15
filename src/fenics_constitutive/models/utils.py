from __future__ import annotations

import dolfinx as df
import numpy as np

from .interfaces import IncrSmallStrainModel, StressStrainConstraint

__all__ = [
    "lame_parameters",
    "get_elastic_tangent",
    "get_identity",
    "strain_from_grad_u",
    "UniaxialStrainFrom3D",
    "PlaneStrainFrom3D",
]


def lame_parameters(E: float, nu: float) -> tuple[float, float]:
    """Compute Lame parameters (mu, lam) from Young's modulus E and Poisson's ratio nu."""
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def get_elastic_tangent(
    E: float, nu: float, constraint: StressStrainConstraint
) -> np.ndarray:
    """Get the linear elastic tangent based on the stress-strain constraint.
    Args:
        E (float): Young's modulus.
        nu (float): Poisson's ratio.
        constraint (StressStrainConstraint): Stress-strain constraint (.FULL, PLANE_STRAIN, PLANE_STRESS, UNIAXIAL_STRAIN, UNIAXIAL_STRESS).

    Returns:
        np.ndarray: Elastic tangent matrix.
    """

    mu, lam = lame_parameters(E, nu)
    match constraint:
        case StressStrainConstraint.FULL:
            # see https://en.wikipedia.org/wiki/Hooke%27s_law
            D = np.array(
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
            D = np.array(
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
            D = (
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
            D = np.array([[C]])

        case StressStrainConstraint.UNIAXIAL_STRESS:
            # see https://csmbrannon.net/2012/08/02/distinction-between-uniaxial-stress-and-uniaxial-strain/
            D = np.array([[E]])

        case _:
            msg = "Constraint not implemented"
            raise NotImplementedError(msg)

    return D


def get_identity(
    stress_strain_dim: int, constraint: StressStrainConstraint
) -> np.ndarray:
    """Get the identity tensor based on the stress-strain constraint.
    Args:
        stress_strain_dim (int): Dimension of the stress/strain vector.
        constraint (StressStrainConstraint): Stress-strain constraint (.FULL, PLANE_STRAIN, PLANE_STRESS, UNIAXIAL_STRAIN, UNIAXIAL_STRESS)
    Returns:
        np.ndarray: Identity tensor.
    """

    match constraint:
        case StressStrainConstraint.FULL:
            I2 = np.zeros(stress_strain_dim, dtype=np.float64)
            I2[0:3] = 1.0
        case StressStrainConstraint.PLANE_STRAIN:
            # We assert that the strain is being provided with 0 in the z-direction
            I2 = np.zeros(stress_strain_dim, dtype=np.float64)
            I2[0:3] = 1.0
        case StressStrainConstraint.PLANE_STRESS:
            I2 = np.zeros(stress_strain_dim, dtype=np.float64)
            I2[0:2] = 1.0
        case StressStrainConstraint.UNIAXIAL_STRAIN:
            I2 = np.zeros(stress_strain_dim, dtype=np.float64)
            I2[0] = 1.0
        case StressStrainConstraint.UNIAXIAL_STRESS:
            I2 = np.zeros(stress_strain_dim, dtype=np.float64)
            I2[0] = 1.0

        case _:
            msg = "Constraint not implemented"
            raise NotImplementedError(msg)

    return I2


def strain_from_grad_u(
    grad_u: np.ndarray, constraint: StressStrainConstraint
) -> np.ndarray:
    """
    Compute the strain in Mandel notation from the gradient of displacement (or increments of both
    quantities).

    Args:
        grad_u: Gradient of displacement field.
        constraint: Constraint that the model is implemented for.

    Returns:
        Numpy array containing the strain for all IPs.
    """
    strain_dim = constraint.stress_strain_dim
    n_gauss = int(grad_u.size / (constraint.geometric_dim**2))
    strain = np.zeros(strain_dim * n_gauss)
    grad_u_view = grad_u.reshape(-1, constraint.geometric_dim**2)
    strain_view = strain.reshape(-1, strain_dim)

    match constraint:
        case StressStrainConstraint.UNIAXIAL_STRAIN:
            strain_view[:, 0] = grad_u_view[:, 0]
        case StressStrainConstraint.UNIAXIAL_STRESS:
            strain_view[:, 0] = grad_u_view[:, 0]
        case StressStrainConstraint.PLANE_STRAIN:
            """
            Full tensor:

            0 1
            2 3

            Mandel vector:
            f = 1 / sqrt(2)
            0 3 "0" f*(1+2)
            """
            strain_view[:, 0] = grad_u_view[:, 0]
            strain_view[:, 1] = grad_u_view[:, 3]
            strain_view[:, 2] = 0.0
            strain_view[:, 3] = 1 / 2**0.5 * (grad_u_view[:, 1] + grad_u_view[:, 2])
        case StressStrainConstraint.PLANE_STRESS:
            """
            Full tensor:

            0 1
            2 3

            Mandel vector:
            f = 1 / sqrt(2)
            0 3 "0" f*(1+2)
            """
            strain_view[:, 0] = grad_u_view[:, 0]
            strain_view[:, 1] = grad_u_view[:, 3]
            strain_view[:, 2] = 0.0
            strain_view[:, 3] = 1 / 2**0.5 * (grad_u_view[:, 1] + grad_u_view[:, 2])
        case StressStrainConstraint.FULL:
            """
            Full tensor:

            0 1 2
            3 4 5
            6 7 8

            Mandel vector:
            f = 1 / sqrt(2)
            0 4 8 f*(1+3) f*(5+7) f*(2+6)
            """
            strain_view[:, 0] = grad_u_view[:, 0]
            strain_view[:, 1] = grad_u_view[:, 4]
            strain_view[:, 2] = grad_u_view[:, 8]
            strain_view[:, 3] = 1 / 2**0.5 * (grad_u_view[:, 1] + grad_u_view[:, 3])
            strain_view[:, 4] = 1 / 2**0.5 * (grad_u_view[:, 2] + grad_u_view[:, 6])
            strain_view[:, 5] = 1 / 2**0.5 * (grad_u_view[:, 5] + grad_u_view[:, 7])
        case _:
            msg = "Constraint not supported."
            raise NotImplementedError(msg)
    return strain


class UniaxialStrainFrom3D(IncrSmallStrainModel):
    """
    Convert a 3D model to a uniaxial strain model. This is achieved by copying the
    relevant 1D components to a 3D array, calling the 3D model, and then copying the
    3D components back to the 1D array. Since this converter creates new arrays for
    the 3D components, it is (probably) not suitable for large-scale simulations due
    to the memory consumption.

    Args:
        model: 3D model to convert to uniaxial strain.

    Attributes:
        model (IncrSmallStrainModel): 3D model to convert to uniaxial strain.
        stress_3d (np.ndarray): 3D stress array.
        tangent_3d (np.ndarray): 3D tangent array.
        grad_del_u_3d (np.ndarray): 3D array of gradient of displacement increment.
    """

    def __init__(self, model: IncrSmallStrainModel) -> None:
        assert model.constraint == StressStrainConstraint.FULL
        self.model = model
        self.stress_3d = None
        self.tangent_3d = None
        self.grad_del_u_3d = None

    @property
    def constraint(self) -> StressStrainConstraint:
        return StressStrainConstraint.UNIAXIAL_STRAIN

    def update(self) -> None:
        self.model.update()

    def evaluate(
        self,
        time: float,
        del_t: float,
        grad_del_u: np.ndarray,
        mandel_stress: np.ndarray,
        tangent: np.ndarray,
        history: np.ndarray | dict[str, np.ndarray] | None,
    ) -> None:
        self.tangent_3d = (
            np.zeros(6 * 6 * len(grad_del_u))
            if self.tangent_3d is None
            else self.tangent_3d
        )
        self.stress_3d = (
            np.zeros(6 * len(grad_del_u)) if self.stress_3d is None else self.stress_3d
        )
        self.grad_del_u_3d = (
            np.zeros(9 * len(grad_del_u))
            if self.grad_del_u_3d is None
            else self.grad_del_u_3d
        )

        self._grad_del_u_to_3d(grad_del_u)
        self._stress_to_3d(mandel_stress)

        self.model.evaluate(
            time, del_t, self.grad_del_u_3d, self.stress_3d, self.tangent_3d, history
        )
        self._tangent_to_1d(tangent)
        self._stress_to_1d(mandel_stress)

    @property
    def history_dim(self) -> dict[str, int | tuple[int, int]] | None:
        return self.model.history_dim

    @df.common.timed("model-conversion-wrapper")
    def _grad_del_u_to_3d(self, grad_del_u_1d: np.ndarray) -> None:
        # Copy the 11 component of the 1D grad_del_u to the 3D grad_del_u
        self.grad_del_u_3d.reshape(-1, 9)[:, 0] = grad_del_u_1d

    @df.common.timed("constitutive-model-conversion-wrapper")
    def _stress_to_3d(self, stress_1d: np.ndarray) -> None:
        # Copy the 11 component of the 1D stress to the 3D stress
        self.stress_3d.reshape(-1, 6)[:, 0] = stress_1d

    @df.common.timed("model-conversion-wrapper")
    def _stress_to_1d(self, stress_1d: np.ndarray) -> None:
        # Copy the 11 component of the 3D stress to the 1D stress
        stress_1d[:] = self.stress_3d.reshape(-1, 6)[:, 0]

    @df.common.timed("model-conversion-wrapper")
    def _tangent_to_1d(self, tangent_1d: np.ndarray) -> None:
        # Copy the 11 component of the 3D tangent to the 1D tangent
        tangent_1d[:] = self.tangent_3d.reshape(-1, 36)[:, 0]


class PlaneStrainFrom3D(IncrSmallStrainModel):
    """
    Convert a 3D model to a plane strain model. This is achieved by copying the
    relevant 2D components to a 3D array, calling the 3D model, and then copying the
    3D components back to the 2D array. Since this converter creates new arrays for
    the 3D components, it is (probably) not suitable for large-scale simulations due
    to the memory consumption.

    Args:
        model: 3D model to convert to plane strain.

    Attributes:
        model (IncrSmallStrainModel): 3D model to convert to plane strain.
        stress_3d (np.ndarray): 3D stress array.
        tangent_3d (np.ndarray): 3D tangent array.
        grad_del_u_3d (np.ndarray): 3D array of gradient of displacement increment.
    """

    def __init__(self, model: IncrSmallStrainModel) -> None:
        assert model.constraint == StressStrainConstraint.FULL
        self.model = model
        self.stress_3d = None
        self.tangent_3d = None
        self.grad_del_u_3d = None

    @property
    def constraint(self) -> StressStrainConstraint:
        return StressStrainConstraint.PLANE_STRAIN

    def update(self) -> None:
        self.model.update()

    def evaluate(
        self,
        time: float,
        del_t: float,
        grad_del_u: np.ndarray,
        mandel_stress: np.ndarray,
        tangent: np.ndarray,
        history: np.ndarray | dict[str, np.ndarray] | None,
    ) -> None:
        n_gauss = int(grad_del_u.size / 4)
        self.tangent_3d = (
            np.zeros(6 * 6 * n_gauss) if self.tangent_3d is None else self.tangent_3d
        )
        self.stress_3d = (
            np.zeros(6 * n_gauss) if self.stress_3d is None else self.stress_3d
        )
        self.grad_del_u_3d = (
            np.zeros(9 * n_gauss) if self.grad_del_u_3d is None else self.grad_del_u_3d
        )

        self._grad_del_u_to_3d(grad_del_u)
        self._stress_to_3d(mandel_stress)

        self.model.evaluate(
            time, del_t, self.grad_del_u_3d, self.stress_3d, self.tangent_3d, history
        )
        self._tangent_to_2d(tangent)
        self._stress_to_2d(mandel_stress)

    @property
    def history_dim(self) -> dict[str, int | tuple[int, int]] | None:
        return self.model.history_dim

    @df.common.timed("model-conversion-wrapper")
    def _grad_del_u_to_3d(
        self,
        grad_del_u_2d: np.ndarray,
    ) -> None:
        # grad_del_u_2d: 0 1
        #                2 3
        # grad_del_u_3d: 0 1 2
        #                3 4 5
        #                6 7 8
        # We map the 0, 1, 2,3 components of the 2D grad_del_u to the 3D grad_del_u
        # components 0, 1, 3, 4
        self.grad_del_u_3d.reshape(-1, 9)[:, 0:2] = grad_del_u_2d.reshape(-1, 4)[:, 0:2]
        self.grad_del_u_3d.reshape(-1, 9)[:, 3:5] = grad_del_u_2d.reshape(-1, 4)[:, 2:4]

    @df.common.timed("model-conversion-wrapper")
    def _stress_to_3d(
        self,
        stress_2d: np.ndarray,
    ) -> None:
        # Copy the 0, 1, 2, 3 components of the 2D stress to the 3D stress
        self.stress_3d.reshape(-1, 6)[:, 0:4] = stress_2d.reshape(-1, 4)

    @df.common.timed("model-conversion-wrapper")
    def _stress_to_2d(self, stress_2d: np.ndarray) -> None:
        # Copy the 0, 1, 2, 3 components of the 3D stress to the 2D stress
        stress_2d.reshape(-1, 4)[:] = self.stress_3d.reshape(-1, 6)[:, 0:4]

    @df.common.timed("model-conversion-wrapper")
    def _tangent_to_2d(self, tangent_2d: np.ndarray) -> None:
        # tangent_2d: 0 1 2 3
        #             4 5 6 7
        #             8 9 10 11
        #             12 13 14 15
        # tangent_3d: 0 1 2 3 4 5
        #             6 7 8 9 10 11
        #             12 13 14 15 16 17
        #             18 19 20 21 22 23
        #             24 25 26 27 28 29
        #             30 31 32 33 34 35
        # We map the first 4x4 block of the 3D tangent to the 2D tangent
        view_2d = tangent_2d.reshape(-1, 16)
        view_3d = self.tangent_3d.reshape(-1, 36)

        view_2d[:, 0:4] = view_3d[:, 0:4]
        view_2d[:, 4:8] = view_3d[:, 6:10]
        view_2d[:, 8:12] = view_3d[:, 12:16]
        view_2d[:, 12:16] = view_3d[:, 18:22]
