
from __future__ import annotations
import dolfinx as df
from petsc4py import PETSc
from ._solver import IncrSmallStrainProblem
from .corotational_lawonsubmesh import CorotationalLawOnSubMesh
from dolfinx.fem.petsc import NonlinearProblem

class CDMSolver(IncrSmallStrainProblem):
    """
    Corotational extension of IncrSmallStrainProblem that adds objective
    stress-rate rotation and mesh-update steps.

    This subclass reuses all core functionality of the base solver and
    overrides only the initialization and `form` method.
    """

    def __init__(self, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

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


