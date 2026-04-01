from __future__ import annotations

from dataclasses import dataclass

from fenics_constitutive.solver._incrementalunknowns import IncrementalDisplacement


@dataclass
class MeshUpdater:
    """
    Class to update the mesh geometry based on the current and previous displacements.
    It assumes that the elements are isoparametric, i.e. the geometry description has
    the same order as the shape functions.

    Args:
        displacements: The incremental displacements containing the current and previous displacements.

    Attributes:
        x_initial: np.ndarray
        mesh: df.mesh.Mesh
        displacements: IncrementalDisplacement
    """

    def __init__(self, displacements: IncrementalDisplacement):
        self._check_isoparametric(displacements)
        self.displacements = displacements
        self.mesh = self.displacements.u.function_space.mesh
        self.x_initial = self.mesh.geometry.x.copy()

    def move_to_midpoint(self):
        """Move the mesh to the midpoint between the current and previous configuration."""
        self.move_to_generalized_midpoint(0.5)

    def move_to_generalized_midpoint(self, alpha: float):
        r"""
        Move the mesh to the generalized midpoint $x_{n+\alpha} = x_n + \alpha \Delta x$
        between the current and previous configuration.
        Args:
            alpha: The parameter $\alpha$ in the generalized midpoint. Should be between 0 and 1.
        """
        current = self.displacements.current.x.array
        previous = self.displacements.previous.x.array
        delta_u = current - previous

        # Update the reference configuration to midpoint
        self.mesh.geometry.x[:] = self.x_initial + (previous + alpha * delta_u).reshape(
            -1, 3
        )

    def move_to_final(self):
        r"""
        Move the mesh to the final configuration $X_{n+1}$.
        """
        current = self.displacements.current.x.array

        # Update the reference configuration to midpoint
        self.mesh.geometry.x[:] = self.x_initial + current.reshape(-1, 3)

    def move_to_previous(self):
        previous = self.displacements.previous.x.array

        # Update the reference configuration to midpoint
        self.mesh.geometry.x[:] = self.x_initial + previous.reshape(-1, 3)

    def move_to_initial(self):
        r"""
        Move the mesh to the initial configuration $x_0$.
        """
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
