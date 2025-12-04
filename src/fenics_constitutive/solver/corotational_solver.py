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

        # Reference configuration
        self._X_ref_debug = self._u.function_space.mesh.geometry.x.copy()

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
        self._move_mesh_previous_to_midpoint()

        NonlinearProblem.form(self, x)
        self.incr_disp.update_current(x)

        for law in self._law_on_submeshs:
            law.evaluate(self.sim_time, self.incr_disp, self.stress, self.tangent)

        self._move_mesh_midpoint_to_final()

        self.stress.scatter_current()
        self.tangent.x.scatter_forward()

    def _move_mesh_previous_to_midpoint(self):
        """Moves mesh geometry to midpoint configuration."""

        V_CG = df.fem.functionspace(self._u.function_space.mesh, ("CG", 1, (3,)))
        u_CG0 = df.fem.Function(V_CG)
        u_CG = df.fem.Function(V_CG)

        # interpolate displacements to CG-1
        # interpolation necessary for higher order elements
        u_CG0.interpolate(self._u0)
        u_CG.interpolate(self._u)

        # displacement increment needed to move to midpoint
        self._midpoint_disp = 0.5 * (u_CG.x.array - u_CG0.x.array)

        # Update the reference configuration to midpoint
        mesh = self._u.function_space.mesh
        mesh.geometry.x[:] = self._X_ref_debug + u_CG0.x.array.reshape(-1, 3) +  self._midpoint_disp.reshape(-1, 3)

    def _move_mesh_midpoint_to_final(self):
        """Moves mesh geometry from midpoint to final configuration."""

        # Update the midpoint to final configuration
        mesh = self._u.function_space.mesh
        mesh.geometry.x[:] += self._midpoint_disp.reshape(-1, 3)




