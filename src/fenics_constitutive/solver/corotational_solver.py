from __future__ import annotations
import dolfinx as df
from petsc4py import PETSc
from ._solver import IncrSmallStrainProblem
from .corotational_lawonsubmesh import CorotationalLawOnSubMesh
from dolfinx.fem.petsc import NonlinearProblem

class CorotationalIncrSmallStrainProblem(IncrSmallStrainProblem):
    """
    Corotational extension of IncrSmallStrainProblem that adds objective
    stress-rate rotation and mesh-update steps.

    This subclass reuses all core functionality of the base solver and
    overrides only the initialization and `form` method.
    """

    def __init__(self, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self._check_spatial_dimension_3d()
        self._check_isoparametric()

        # Reference configuration
        self._X_ref = self._u.function_space.mesh.geometry.x.copy()

        # Replace each base LawOnSubMesh with its corotational variant
        # while preserving all submesh data and function references
        self._law_on_submeshs = [
            CorotationalLawOnSubMesh(
                law=lom.law,
                cells=lom.cells,
                displacement_gradient_fn=lom.displacement_gradient_fn,
                stress=lom.stress,
                local_tangent=lom.local_tangent,
                submesh_map=lom.submesh_map,
                history=lom.history,
            )
            for lom in self._law_on_submeshs
        ]

    @df.common.timed("constitutive-form-evaluation-corotational")
    def form(self, x: PETSc.Vec) -> None:
        """
        Assemble the nonlinear form using a corotational midpoint configuration.

        This override performs a mesh update to the midpoint configuration
        and applies a corotational stress rotation before evaluating the constitutive
        laws. The mesh is updated to final configuration prior to scattering stress and tangent fields.
        """
        NonlinearProblem.form(self, x)

        self._move_mesh_previous_to_midpoint()

        self.incr_disp.update_current(x)

        for law in self._law_on_submeshs:
            law.evaluate(self.sim_time, self.incr_disp, self.stress, self.tangent)

        self._move_mesh_midpoint_to_final()

        self.stress.scatter_current()
        self.tangent.x.scatter_forward()

    def _move_mesh_previous_to_midpoint(self):
        """Moves mesh geometry to midpoint configuration."""

        # displacement increment needed to move to midpoint
        midpoint_disp = 0.5 * (self._u.x.array - self._u0.x.array)

        # Update the reference configuration to midpoint
        mesh = self._u.function_space.mesh
        mesh.geometry.x[:] = self._X_ref + self._u0.x.array.reshape(-1, 3) +  midpoint_disp.reshape(-1, 3)

    def _move_mesh_midpoint_to_final(self):
        """Moves mesh geometry from midpoint to final configuration."""

        # Update the midpoint to final configuration
        mesh = self._u.function_space.mesh
        mesh.geometry.x[:] = self._X_ref + self._u.x.array.reshape(-1, 3)

    def _check_isoparametric(self):
        """Checks if the elements are isoparametric."""

        # Geometry degree (e.g. 1 for linear, 2 for quadratic)
        geom_degree = self._u.function_space.mesh.geometry.cmap.degree

        # Displacement element degree
        disp_degree = self._u.function_space.ufl_element().degree

        if geom_degree != disp_degree:
            raise NotImplementedError(
                f"Mesh update only supported for isoparametric elements: "
                f"geometry degree {geom_degree}, displacement degree {disp_degree}"
            )

    def _check_spatial_dimension_3d(self) -> None:
        """
        Ensure that the solver is only used for 3D problems.

        The current corotational implementation assumes 3x3 tensors and a
        6-component Mandel stress (3 normal + 3 shear). For 2D / 1D
        constitutive models a separate implementation is required.
        """
        gdim = self._u.function_space.mesh.geometry.dim
        if gdim != 3:
            raise NotImplementedError(
                f"CorotationalIncrSmallStrainProblem currently supports only 3D "
                f"geometries (gdim=3). Got gdim={gdim}."
            )