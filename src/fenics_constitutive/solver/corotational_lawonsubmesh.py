from dataclasses import dataclass

import dolfinx as df
import numpy as np
from scipy.linalg import expm, logm

from ._lawonsubmesh import IncrementalDisplacement, IncrementalStress, LawOnSubMesh
from ._solver import SimulationTime


@dataclass
class CorotationalLawOnSubMesh(LawOnSubMesh):
    """LawOnSubMesh with corotational stress rotation."""

    def evaluate(
        self,
        sim_time: SimulationTime,
        incr_disp: IncrementalDisplacement,
        global_stress: IncrementalStress,
        global_tangent: df.fem.Function,
    ) -> None:

        incr_disp.evaluate_local_incremental_gradient(
            self.cells, self.displacement_gradient_fn
        )
        history_input = (
            self.history.reset_trial_state() if self.history is not None else None
        )

        # get local Mandel stress array once
        local_stress_arr = self.local_stress(global_stress)  # self.stress.x.array

        # calculate half rotation matrix for all quadrature points
        Q_half_all = self._compute_half_rotations(self.displacement_gradient_fn.x.array)

        # rotate to midpoint before passing to constitutive law (half rotation)
        self._stress_rotate(Q_half_all, local_stress_arr)

        with df.common.Timer("constitutive-law-evaluation"):
            self.law.evaluate(
                sim_time.current,
                sim_time.dt,
                self.displacement_gradient_fn.x.array,
                local_stress_arr,
                self.local_tangent.x.array,
                history_input,
            )

        # rotate to final configuration (half rotation)
        self._stress_rotate(Q_half_all, local_stress_arr)
        self.map_to_parent(global_stress, global_tangent)

    def _stress_rotate(self, Q_half_all: np.ndarray, mandel_stress: np.ndarray) -> None:
        """Rotate Mandel stress using precomputed half-step rotations Q_half_all."""

        n = Q_half_all.shape[0]
        mandel_stress = mandel_stress.reshape(-1, 6)

        # build full 3x3 stress tensor from Mandel representation
        stress = np.zeros((n, 3, 3), dtype=np.float64)
        stress[:, 0, 0] = mandel_stress[:, 0]
        stress[:, 1, 1] = mandel_stress[:, 1]
        stress[:, 2, 2] = mandel_stress[:, 2]
        stress[:, 0, 1] = 1.0 / np.sqrt(2.0) * mandel_stress[:, 3]
        stress[:, 1, 2] = 1.0 / np.sqrt(2.0) * mandel_stress[:, 4]
        stress[:, 0, 2] = 1.0 / np.sqrt(2.0) * mandel_stress[:, 5]
        stress[:, 1, 0] = stress[:, 0, 1]
        stress[:, 2, 1] = stress[:, 1, 2]
        stress[:, 2, 0] = stress[:, 0, 2]

        for i in range(n):
            Q_half = Q_half_all[i]
            stress[i, :, :] = Q_half.T @ stress[i, :, :] @ Q_half

        # back to Mandel
        rotated_stress_mandel = np.zeros((n, 6), dtype=np.float64)
        rotated_stress_mandel[:, 0] = stress[:, 0, 0]
        rotated_stress_mandel[:, 1] = stress[:, 1, 1]
        rotated_stress_mandel[:, 2] = stress[:, 2, 2]
        rotated_stress_mandel[:, 3] = np.sqrt(2.0) * stress[:, 0, 1]
        rotated_stress_mandel[:, 4] = np.sqrt(2.0) * stress[:, 1, 2]
        rotated_stress_mandel[:, 5] = np.sqrt(2.0) * stress[:, 0, 2]

        # write back in-place
        mandel_stress[:, :] = rotated_stress_mandel

    def _compute_half_rotations(self, del_grad_u: np.ndarray) -> np.ndarray:
        """
        Compute half-step rotation matrices Q_half for all quadrature points.

        Returns:
            Q_half: array of shape (N, 3, 3)
        """
        I2 = np.eye(3)
        n = del_grad_u.size // 9
        g = del_grad_u.reshape(n, 3, 3)

        Q_half_all = np.zeros((n, 3, 3))
        for i, eps in enumerate(g):
            rotation_increment = 0.5 * (eps - eps.T)
            Q_matrix = I2 + np.linalg.inv(I2 - 0.5 * rotation_increment) @ rotation_increment
            log_Q = logm(Q_matrix)
            Q_half_all[i] = expm(0.5 * log_Q)
        return Q_half_all
