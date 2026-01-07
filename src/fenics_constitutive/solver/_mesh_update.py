from __future__ import annotations

from dataclasses import dataclass

import dolfinx as df
import numpy as np

from fenics_constitutive.solver._incrementalunknowns import IncrementalDisplacement


@dataclass
class MeshUpdater:
    x_initial: np.ndarray
    mesh: df.mesh.Mesh
    displacements: IncrementalDisplacement

    def __init__(self, displacements: IncrementalDisplacement):
        self._check_isoparametric(displacements)
        self.displacements = displacements
        self.mesh = self.displacements.u.function_space.mesh
        self.x_initial = self.mesh.geometry.x.copy()

    def move_to_midpoint(self):
        self.move_to_generalized_midpoint(0.5)

    def move_to_generalized_midpoint(self, alpha: float):
        current = self.displacements.current.x.array
        previous = self.displacements.previous.x.array
        delta_u = current - previous

        # Update the reference configuration to midpoint
        self.mesh.geometry.x[:] = self.x_initial + (previous + alpha * delta_u).reshape(
            -1, 3
        )

    def move_to_final(self):
        current = self.displacements.current.x.array

        # Update the reference configuration to midpoint
        self.mesh.geometry.x[:] = self.x_initial + current.reshape(-1, 3)

    def move_to_previous(self):
        previous = self.displacements.previous.x.array

        # Update the reference configuration to midpoint
        self.mesh.geometry.x[:] = self.x_initial + previous.reshape(-1, 3)

    def move_to_initial(self):
        self.mesh.geometry.x[:] = self.x_initial

    def _check_isoparametric(self, d: IncrementalDisplacement):
        """Checks if the elements are isoparametric."""

        # Geometry degree (e.g. 1 for linear, 2 for quadratic)
        geom_degree = d.u.function_space.mesh.geometry.cmap.degree

        # Displacement element degree
        disp_degree = d.u.function_space.ufl_element().degree

        if geom_degree != disp_degree:
            msg = (
                f"Mesh update only supported for isoparametric elements: "
                f"geometry degree {geom_degree}, displacement degree {disp_degree}"
            )
            raise NotImplementedError(msg)
