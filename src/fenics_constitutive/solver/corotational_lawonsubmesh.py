from dataclasses import dataclass
import numpy as np
from scipy.linalg import logm, expm
from ._lawonsubmesh import LawOnSubMesh, IncrementalDisplacement, IncrementalStress
from ._solver import SimulationTime
import dolfinx as df

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

        # rotate to midpoint before passing to constitutive law (half rotation)
        self._stress_rotate(self.displacement_gradient_fn.x.array, local_stress_arr)

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
        self._stress_rotate(self.displacement_gradient_fn.x.array, local_stress_arr)
        self.map_to_parent(global_stress, global_tangent)

    def _stress_rotate(self, del_grad_u: np.ndarray, mandel_stress: np.ndarray) -> None:
        """
        Rotate Mandel stress using the incremental displacement gradient del_grad_u.

        del_grad_u: flattened array of shape (N*9,)
        mandel_stress: flattened Mandel stress array of shape (N*6,) or (N,6)
                       (modified in-place)
        """
        I2 = np.eye(3)

        # number of quadrature points / entries
        n = del_grad_u.size // 9

        # reshape stresses into Mandel 6-vector
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

        # reshape incremental displacement gradient
        g = del_grad_u.reshape(n, 3, 3)

        for i, eps in enumerate(g):
            # skew part -> incremental rotation
            rotation_increment = 0.5 * (eps - eps.T)

            # incremental rotation matrix Q
            Q_matrix = I2 + np.linalg.inv(I2 - 0.5 * rotation_increment) @ rotation_increment

            # compute half-step rotation via matrix log/exp
            log_Q = logm(Q_matrix)
            log_Q_half = 0.5 * log_Q
            Q_half = expm(log_Q_half)

            # rotate stress
            rot_stress = Q_half.T @ stress[i, :, :] @ Q_half
            stress[i, :, :] = rot_stress

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