from __future__ import annotations

import numpy as np

from fenics_constitutive import Constraint, IncrSmallStrainModel, strain_from_grad_u


class LinearElasticityModel(IncrSmallStrainModel):
    def __init__(self, parameters: dict[str, float], constraint: Constraint):
        self._constraint = constraint
        E = parameters["E"]
        nu = parameters["nu"]
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        match constraint:
            case Constraint.FULL:
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
            case Constraint.PLANE_STRAIN:
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
            case Constraint.PLANE_STRESS:
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
            case Constraint.UNIAXIAL_STRAIN:
                # see https://csmbrannon.net/2012/08/02/distinction-between-uniaxial-stress-and-uniaxial-strain/
                C = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
                self.D = np.array([[C]])
            case Constraint.UNIAXIAL_STRESS:
                # see https://csmbrannon.net/2012/08/02/distinction-between-uniaxial-stress-and-uniaxial-strain/
                self.D = np.array([[E]])
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
        n_gauss = grad_del_u.size // (self.geometric_dim**2)
        mandel_view = mandel_stress.reshape(-1, self.stress_strain_dim)

###################################################################
        # rot_matrix = np.array([
        #     [np.cos(np.pi / 8), np.sin(np.pi / 8), 0],
        #     [-np.sin(np.pi / 8), np.cos(np.pi / 8), 0],
        #     [0, 0, 1]
        # ])
        I2 = np.eye(3, 3)
        shape = int(np.shape(grad_del_u)[0] / 9)
        #
        # g = grad_del_u.reshape(shape, 3, 3)
        # strains = g
        #
        # for n, eps in enumerate(g):
        #     rotation_increment = (eps - np.transpose(eps)) / 2
        #     Q_matrix = I2 + (np.linalg.inv(I2 - 0.5*rotation_increment)) @ rotation_increment
        #     # print(Q_matrix)
        #     strains_rotated = strains[n, :, :]
        #     # print(strains_rotated)
        #     strains[n, :, :] = strains_rotated
        #     # print(strains[n, :, :] - g[n,:,:])
        #
        # # grad_del_u[:] = strains.flatten()
####################################################################################
        strain_increment = strain_from_grad_u(grad_del_u, self.constraint)
        # # print(strain_increment.reshape(-1, self.stress_strain_dim))
        #
        # strain_increment = strain_increment.reshape(-1, 6)
        # strain = np.zeros((shape, 3, 3), dtype=np.float64)
        #
        # strain[:, 0, 0] = strain_increment[:, 0]
        # strain[:, 1, 1] = strain_increment[:, 1]
        # strain[:, 2, 2] = strain_increment[:, 2]
        # strain[:, 0, 1] = 1 / 2 ** 0.5 * (strain_increment[:, 3])
        # strain[:, 1, 2] = 1 / 2 ** 0.5 * (strain_increment[:, 4])
        # strain[:, 0, 2] = 1 / 2 ** 0.5 * (strain_increment[:, 5])
        # strain[:, 1, 0] = strain[:, 0, 1]
        # strain[:, 2, 1] = strain[:, 1, 2]
        # strain[:, 2, 0] = strain[:, 0, 2]
        #
        # g = grad_del_u.reshape(shape, 3, 3)
        #
        # for n, eps in enumerate(g):
        #     rotation_increment = (eps - np.transpose(eps)) / 2
        #     Q_matrix = I2 + (np.linalg.inv(I2 - 0.5 * rotation_increment)) @ rotation_increment
        #     theta = np.arctan2(Q_matrix[1, 0], Q_matrix[0, 0])
        #
        #     # Calculate half angle rotation matrix
        #     Q_matrix_half_angle = np.array([
        #         [np.cos(theta / 2), -np.sin(theta / 2), 0],
        #         [np.sin(theta / 2), np.cos(theta / 2), 0],
        #         [0, 0, 1]
        #     ])
        #
        #     rot_strain = strain[n, :, :]
        #     print(rot_strain - strain[n, :, :])
        #     # print(rotation_increment)
        #     strain[n, :, :] = rot_strain
        #
        # rotated_strain_mandel = np.zeros((shape, 6), dtype=np.float64)
        #
        # rotated_strain_mandel[:, 0] = strain[:, 0, 0]
        # rotated_strain_mandel[:, 1] = strain[:, 1, 1]
        # rotated_strain_mandel[:, 2] = strain[:, 2, 2]
        # rotated_strain_mandel[:, 3] = 2 ** 0.5 * strain[:, 0, 1]
        # rotated_strain_mandel[:, 4] = 2 ** 0.5 * strain[:, 1, 2]
        # rotated_strain_mandel[:, 5] = 2 ** 0.5 * strain[:, 0, 2]
        #
        # # print('mandel stress rotated ################',rotated_stress_mandel)
        # # mandel_stress = mandel_stress.flatten()
        # strain_increment[:, :] = rotated_strain_mandel
        # strain_increment.flatten()

        mandel_view += strain_increment.reshape(-1, self.stress_strain_dim) @ self.D

######################################################################################
        # I2 = np.eye(3, 3)
        #
        # str_co = str_co.reshape(-1, 6)
        #
        # stress = np.zeros((shape, 3, 3), dtype=np.float64)
        #
        # stress[:, 0, 0] = str_co[:, 0]
        # stress[:, 1, 1] = str_co[:, 1]
        # stress[:, 2, 2] = str_co[:, 2]
        # stress[:, 0, 1] = 1 / 2 ** 0.5 * (str_co[:, 3])
        # stress[:, 1, 2] = 1 / 2 ** 0.5 * (str_co[:, 4])
        # stress[:, 0, 2] = 1 / 2 ** 0.5 * (str_co[:, 5])
        # stress[:, 1, 0] = stress[:, 0, 1]
        # stress[:, 2, 1] = stress[:, 1, 2]
        # stress[:, 2, 0] = stress[:, 0, 2]
        #
        #
        # g = grad_del_u.reshape(shape, 3, 3)
        #
        #
        # for n, eps in enumerate(g):
        #     rotation_increment = (eps - np.transpose(eps)) / 2
        #     Q_matrix = I2 + (np.linalg.inv(I2 - 0.5 * rotation_increment)) @ rotation_increment
        #     theta = np.arctan2(Q_matrix[1, 0], Q_matrix[0, 0])
        #
        #     # Calculate half angle rotation matrix
        #     Q_matrix_half_angle = np.array([
        #         [np.cos(theta / 2), -np.sin(theta / 2), 0],
        #         [np.sin(theta / 2), np.cos(theta / 2), 0],
        #         [0, 0, 1]
        #     ])
        #     # Q_matrix = I2 + (np.linalg.inv(I2 - 0.5 * rotation_increment)) @ rotation_increment
        #     rot_stress = stress[n, :, :]
        #     # print(strains_rotated-eps)
        #     # print(rotation_increment)
        #     stress[n, :, :] = rot_stress
        #     # rotated_stress_matrix.append(rot_stress)
        #
        # # rotated_stress_matrix = np.array(rotated_stress_matrix)
        # # print(np.shape(rotated_stress_matrix))
        # rotated_stress_mandel = np.zeros((shape, 6), dtype=np.float64)
        #
        # rotated_stress_mandel[:, 0] = stress[:, 0, 0]
        # rotated_stress_mandel[:, 1] = stress[:, 1, 1]
        # rotated_stress_mandel[:, 2] = stress[:, 2, 2]
        # rotated_stress_mandel[:, 3] = 2 ** 0.5 * stress[:, 0, 1]
        # rotated_stress_mandel[:, 4] = 2 ** 0.5 * stress[:, 1, 2]
        # rotated_stress_mandel[:, 5] = 2 ** 0.5 * stress[:, 0, 2]
        #
        # # print('mandel stress rotated ################',rotated_stress_mandel)
        # # mandel_stress = mandel_stress.flatten()
        # # str_co = rotated_stress_mandel
        # rotated_stress_mandel.flatten()
        #
        # mandel_view += rotated_stress_mandel
###############
        mandel_view = mandel_view.reshape(-1, 6)

        stress = np.zeros((shape, 3, 3), dtype=np.float64)

        stress[:, 0, 0] = mandel_view[:, 0]
        stress[:, 1, 1] = mandel_view[:, 1]
        stress[:, 2, 2] = mandel_view[:, 2]
        stress[:, 0, 1] = 1 / 2 ** 0.5 * (mandel_view[:, 3])
        stress[:, 1, 2] = 1 / 2 ** 0.5 * (mandel_view[:, 4])
        stress[:, 0, 2] = 1 / 2 ** 0.5 * (mandel_view[:, 5])
        stress[:, 1, 0] = stress[:, 0, 1]
        stress[:, 2, 1] = stress[:, 1, 2]
        stress[:, 2, 0] = stress[:, 0, 2]

        g = grad_del_u.reshape(shape, 3, 3)

        for n, eps in enumerate(g):
            rotation_increment = (eps - np.transpose(eps)) / 2
            Q_matrix = I2 + (np.linalg.inv(I2 - 0.5 * rotation_increment)) @ rotation_increment
            theta = np.arctan2(Q_matrix[1, 0], Q_matrix[0, 0])

            # Calculate half angle rotation matrix
            Q_matrix_half_angle = np.array([
                [np.cos(theta / 2), -np.sin(theta / 2), 0],
                [np.sin(theta / 2), np.cos(theta / 2), 0],
                [0, 0, 1]
            ])
            # Q_matrix = I2 + (np.linalg.inv(I2 - 0.5 * rotation_increment)) @ rotation_increment
            rot_stress = Q_matrix_half_angle.T @ stress[n, :, :] @ Q_matrix_half_angle
            # print(Q_matrix_half_angle)
            # print(rotation_increment)
            stress[n, :, :] = rot_stress
            # rotated_stress_matrix.append(rot_stress)

        # rotated_stress_matrix = np.array(rotated_stress_matrix)
        # print(np.shape(rotated_stress_matrix))
        rotated_stress_mandel = np.zeros((shape, 6), dtype=np.float64)

        rotated_stress_mandel[:, 0] = stress[:, 0, 0]
        rotated_stress_mandel[:, 1] = stress[:, 1, 1]
        rotated_stress_mandel[:, 2] = stress[:, 2, 2]
        rotated_stress_mandel[:, 3] = 2 ** 0.5 * stress[:, 0, 1]
        rotated_stress_mandel[:, 4] = 2 ** 0.5 * stress[:, 1, 2]
        rotated_stress_mandel[:, 5] = 2 ** 0.5 * stress[:, 0, 2]

        # print('mandel stress rotated ################',rotated_stress_mandel)
        # mandel_stress = mandel_stress.flatten()
        mandel_view[:,:] = rotated_stress_mandel
        mandel_view.flatten()




        tangent[:] = np.tile(self.D.flatten(), n_gauss)

    @property
    def constraint(self) -> Constraint:
        return self._constraint

    @property
    def history_dim(self) -> None:
        return None

    def update(self) -> None:
        pass
