from __future__ import annotations

from dataclasses import dataclass

import basix.ufl
import dolfinx as df
import numpy as np

from fenics_constitutive.interfaces import IncrSmallStrainModel


def build_history(
    history_dim: dict[str, int | tuple[int, int]], mesh: df.mesh.Mesh, q_degree: int
) -> dict[str, df.fem.Function]:
    """Build the history space and function(s) for the given law.

    Args:
        history_dim: A dictionary representing the dimensions of the history.
        mesh: Either the full mesh for a homogenous domain or the submesh.
        q_degree: The quadrature degree.

    Returns:
        The history function(s) for the given law or None if the history dimension is 0.

    """
    history = {}
    for key, value in history_dim.items():
        value_shape = (value,) if isinstance(value, int) else value
        Q = basix.ufl.quadrature_element(
            mesh.topology.cell_name(), value_shape=value_shape, degree=q_degree
        )
        history_space = df.fem.functionspace(mesh, Q)
        history[key] = df.fem.Function(history_space)
    return history


@dataclass
class History:
    """A class to hold the history of a constitutive law."""

    history_0: dict[str, df.fem.Function]
    history_1: dict[str, df.fem.Function]
    _history_dim: dict[str, int | tuple[int, int]]

    @staticmethod
    def try_create(
        law: IncrSmallStrainModel, submesh: df.mesh.Mesh, q_degree: int
    ) -> History | None:
        if law.history_dim is None:
            return None

        history_0 = build_history(law.history_dim, submesh, q_degree)
        history_1 = (
            {key: fn.copy() for key, fn in history_0.items()}
            if isinstance(history_0, dict)
            else history_0
        )
        return History(
            history_0=history_0,
            history_1=history_1,
            _history_dim=law.history_dim,
        )

    def reset_trial_state(self) -> dict[str, np.ndarray]:
        """Advance the history for all keys."""
        return {key: self._reset_trial_state_key(key) for key in self._history_dim}

    def update(self) -> dict[str, np.ndarray]:
        """Commit the history for all keys."""
        return {key: self._update_key(key) for key in self._history_dim}

    def _reset_trial_state_key(self, key: str) -> np.ndarray:
        """
        Advance the history for the given key: copy from previous (history_0) to current (history_1)
        and scatter the result.
        """
        self.history_1[key].x.array[:] = self.history_0[key].x.array
        self.history_1[key].x.scatter_forward()
        return self.history_1[key].x.array

    def _update_key(self, key: str) -> np.ndarray:
        """
        Commit the history for the given key: copy from current (history_1) to previous (history_0)
        and scatter the result.
        """
        self.history_0[key].x.array[:] = self.history_1[key].x.array
        self.history_0[key].x.scatter_forward()
        return self.history_0[key].x.array
