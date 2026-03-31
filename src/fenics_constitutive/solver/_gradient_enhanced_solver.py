from __future__ import annotations

from dataclasses import dataclass

import dolfinx as df
import numpy as np
import ufl
from dolfinx.fem.function import Function
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc

from fenics_constitutive.models.interfaces import (
    IncrSmallStrainGradientModel,
    NonlocalTangentFunctions,
    NonlocalTangents,
)
from fenics_constitutive.solver._incrementalunknowns import (
    IncrementalGradientSolution,
    IncrementalLocalQuantity,
)
from fenics_constitutive.solver._lawonsubmesh import (
    GradientLawOnSubMesh,
    create_gradient_law_on_submesh,
)
from fenics_constitutive.solver._solver import SimulationTime
from fenics_constitutive.solver._spaces import GradientElements

from ._incrementalunknowns import IncrementalDisplacement, IncrementalStress
from ._lawonsubmesh import LawOnSubMesh, create_law_on_submesh
from ._spaces import ElementSpaces
from .typesafe import fn_for
from .utils import ufl_mandel_strain


class IncrSmallStrainGradientProblem(NonlinearProblem):
    """
    A nonlinear problem for incremental small strain models. To be used with
    the dolfinx NewtonSolver.

    Args:
        laws: A list of tuples where the first element is the constitutive law and the second
            element is the cells for the submesh. If only one law is provided, it is assumed
            that the domain is homogenous. The cell indices should be local indices of the MPI-process.
        mixed_solution: The mixed solution function of the displacements and nonlocal quantity. This implementation
            uses a mixed element formulation, instead of a mixed function space.
        bcs: The Dirichlet boundary conditions.
        q_degree: The quadrature degree (Polynomial degree which the quadrature rule needs to integrate exactly).
        l: The internal length scale of the gradient model. Used as $$l^2 \nabla \cdot \nabla$$ in the weak form.
        del_t: The time increment.
        external_forces: The external forces in the weak form. This should be a list of ufl Forms, e.g. for body forces and traction forces.
            We usually assume that the gradient of the nonlocal quantity vnishes on the boundary, so we don't need to add the Neumann boundary
            condition for the nonlocal quantity.
        form_compiler_options: The options for the form compiler.
        jit_options: The options for the JIT compiler.
    """

    def __init__(
        self,
        laws: list[tuple[IncrSmallStrainGradientModel, np.ndarray]]
        | IncrSmallStrainGradientModel,
        mixed_solution: df.fem.Function,
        bcs: list[df.fem.DirichletBC],
        q_degree: int,
        l: float,
        del_t: float = 1.0,
        external_forces: list[ufl.Form] | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
    ) -> None:
        mesh = mixed_solution.function_space.mesh
        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        if isinstance(laws, IncrSmallStrainGradientModel):
            laws = [(laws, np.arange(0, num_cells, dtype=np.int32))]

        constraint = laws[0][0].constraint
        assert all(law[0].constraint == constraint for law in laws), (
            "All laws must have the same constraint"
        )

        elements = GradientElements.create(mesh, constraint, q_degree)
        self.stress = IncrementalStress(elements.stress_space(mesh))
        self.local_quantity = IncrementalLocalQuantity(
            elements.local_quantity_space(mesh)
        )
        self.solution = IncrementalGradientSolution.from_mixed_function(
            mixed_solution, q_degree
        )

        u = self.solution.current.sub(0)
        nonlocal_quantity = self.solution.current.sub(1)

        tangent_spaces = elements.tangent_spaces(mesh)
        self.tangents = NonlocalTangentFunctions(
            dsigma_deps=fn_for(tangent_spaces[0]),
            dsigma_dnonlocal=fn_for(tangent_spaces[1]),
            dlocal_deps=fn_for(tangent_spaces[2]),
            dlocal_dnonlocal=fn_for(tangent_spaces[3]),
        )

        self._law_on_submeshs: list[GradientLawOnSubMesh] = []
        self.sim_time = SimulationTime(dt_max=del_t)

        self._law_on_submeshs = [
            create_gradient_law_on_submesh(
                law, local_cells, elements, self.stress.current.function_space
            )
            for law, local_cells in laws
        ]

        (u_test, nonlocal_test) = ufl.TestFunctions(mixed_solution.function_space)
        (u_trial, nonlocal_trial) = ufl.TrialFunctions(mixed_solution.function_space)

        self.metadata = {"quadrature_degree": q_degree, "quadrature_scheme": "default"}
        self.dxm = ufl.dx(metadata=self.metadata)

        # Define the residual form. The first term comes from the balance of linear momentum
        R_form = (
            ufl.inner(ufl_mandel_strain(u_test, constraint), self.stress.current)
            * self.dxm
        )
        # Add terms containing the nonlocal quantity. 
        R_form += (
            ufl.inner(l**2 * ufl.grad(nonlocal_quantity), ufl.grad(nonlocal_test))
            * self.dxm
        )
        R_form += (
            (nonlocal_quantity - self.local_quantity.current) * nonlocal_test * self.dxm
        )

        if external_forces is not None:
            R_form -= sum(external_forces)

        # Define the Jacobian form. contains the 4 Gateaux derivatives of
        # the residual form.
        dR_form = (
            ufl.inner(
                ufl_mandel_strain(u_trial, constraint),
                ufl.dot(
                    self.tangents.dsigma_deps, ufl_mandel_strain(u_test, constraint)
                ),
            )
            * self.dxm
            + (
                ufl.inner(
                    ufl_mandel_strain(u_test, constraint),
                    self.tangents.dsigma_dnonlocal,
                )
                * nonlocal_trial
            )
            * self.dxm
            - (
                nonlocal_test
                * ufl.inner(
                    self.tangents.dlocal_deps, ufl_mandel_strain(u_trial, constraint)
                )
            )
            * self.dxm
            + l**2
            * ufl.inner(ufl.grad(nonlocal_trial), ufl.grad(nonlocal_test))
            * self.dxm
            + (nonlocal_trial - self.tangents.dlocal_dnonlocal * nonlocal_trial)
            * nonlocal_test
            * self.dxm
        )

        super().__init__(
            R_form,
            mixed_solution,
            bcs=bcs,
            J=dR_form,
            form_compiler_options=form_compiler_options
            if form_compiler_options is not None
            else {},
            jit_options=jit_options if jit_options is not None else {},
        )

    @df.common.timed("constitutive-form-evaluation")
    def form(self, x: PETSc.Vec) -> None:
        """This function is called before the residual or Jacobian is
        computed. This is usually used to update ghost values, but here
        we use it to update the stress, tangent and history.

        Args:
            x: The vector containing the latest solution

        """
        super().form(x)
        self.solution.set_current(x)

        for law in self._law_on_submeshs:
            law.evaluate(
                self.sim_time,
                self.solution,
                self.stress,
                self.local_quantity,
                self.tangents,
            )

        self.stress.scatter_current()  # TODO: this scattering may not be needed because we scatter already in map_to_parent

    def update(self) -> None:
        """
        Update the current displacement, stress and history.
        """
        self.solution.update()
        self.stress.update_previous()
        self.local_quantity.update_previous()
        for law in self._law_on_submeshs:
            law.update_history()

        self.sim_time.advance()

    # -------------------------------------------------------------------
    # NOTE: The following properties are used for backward compatibility
    # -------------------------------------------------------------------

    @property
    def _time(self) -> float:
        return self.sim_time.current

    @_time.setter
    def _time(self, value: float) -> None:
        self.sim_time.current = value

    @property
    def _del_t(self) -> float:
        return self.sim_time.dt

    @_del_t.setter
    def _del_t(self, value: float) -> None:
        self.sim_time.dt = value

    @property
    def _u(self) -> df.fem.Function:
        return self.solution.current.sub(0)

    @property
    def _u0(self) -> df.fem.Function:
        return self.solution.previous.sub(0)

    @property
    def stress_0(self) -> df.fem.Function:
        return self.stress.previous

    @property
    def stress_1(self) -> df.fem.Function:
        return self.stress.current

    @property
    def _history_0(self) -> list[dict[str, Function] | None]:
        """Return a list of history_0 dicts for all laws (for backward compatibility)."""

        def _history_or_none(law) -> dict[str, Function] | None:
            return law.history.history_0 if law.history else None

        return [_history_or_none(law) for law in self._law_on_submeshs]

    @property
    def _history_1(self) -> list[dict[str, Function] | None]:
        """Return a list of history_1 dicts for all laws (for backward compatibility)."""

        def _history_or_none(law) -> dict[str, Function] | None:
            return law.history.history_1 if law.history else None

        return [_history_or_none(law) for law in self._law_on_submeshs]

    @property
    def _del_grad_u(self) -> list[Function]:
        """Return a list of inc_disp_grad Functions for all laws (for backward compatibility)."""
        return [law.displacement_gradient_fn for law in self._law_on_submeshs]
