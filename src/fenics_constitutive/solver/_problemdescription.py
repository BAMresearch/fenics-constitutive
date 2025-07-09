from __future__ import annotations

import itertools

import dolfinx as df
import numpy as np
import ufl
from dolfinx.mesh import MeshTags

from fenics_constitutive.boundarycondition import NeumannBC
from fenics_constitutive.interfaces import IncrSmallStrainModel
from fenics_constitutive.stress_strain import ufl_mandel_strain
from fenics_constitutive.typesafe import fn_for

from ._incrementalunknowns import IncrementalDisplacement, IncrementalStress
from ._lawonsubmesh import LawOnSubMesh
from ._solver import IncrSmallStrainProblem, SimulationTime
from ._spaces import ElementSpaces

ModelScope = tuple[IncrSmallStrainModel, np.ndarray]


class IncrSmallStrainProblemDescription:
    def __init__(
        self,
        *,
        displacement_field: df.fem.Function,
        quadrature_degree: int,
        time_increment: float = 1.0,
        laws: list[ModelScope] | IncrSmallStrainModel,
    ) -> None:
        self.u = displacement_field
        self.q_degree = quadrature_degree
        self.del_t = time_increment

        mesh = self.u.function_space.mesh
        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        global_cells = np.arange(0, num_cells, dtype=np.int32)

        if isinstance(laws, IncrSmallStrainModel):
            self._laws = [(laws, global_cells)]
        else:
            self._laws = laws

        constraint = self._laws[0][0].constraint
        assert all(law[0].constraint == constraint for law in self._laws), (
            "All laws must have the same constraint"
        )

        element_spaces = ElementSpaces.create(mesh, constraint, self.q_degree)
        self.stress = IncrementalStress(element_spaces.stress_vector_space)
        self.tangent = fn_for(element_spaces.stress_tensor_space(mesh))

        self._law_on_submeshs: list[LawOnSubMesh] = []
        self.sim_time = SimulationTime(dt=self.del_t)

        self._law_on_submeshs = [
            LawOnSubMesh.map_to_cells(law, local_cells, element_spaces)
            for law, local_cells in self._laws
        ]

        u_, du = (
            ufl.TestFunction(self.u.function_space),
            ufl.TrialFunction(self.u.function_space),
        )

        self.metadata = {
            "quadrature_degree": self.q_degree,
            "quadrature_scheme": "default",
        }
        self.dxm = ufl.dx(metadata=self.metadata)

        self.R_form = (
            ufl.inner(ufl_mandel_strain(u_, constraint), self.stress.current) * self.dxm
        )
        self.dR_form = (
            ufl.inner(
                ufl_mandel_strain(du, constraint),
                ufl.dot(self.tangent, ufl_mandel_strain(u_, constraint)),
            )
            * self.dxm
        )

        self._dirichlet_bcs = []
        self._neumann_bcs = []
        self._apply_neumann_bcs()

        self.incr_disp = IncrementalDisplacement(self.u, self.q_degree)

    def add_boundary_conditions(self, *bcs: df.fem.DirichletBC | NeumannBC) -> None:
        grouped = {bc: list(group) for bc, group in itertools.groupby(bcs, type)}
        self._dirichlet_bcs.extend(grouped.get(df.fem.DirichletBC, []))
        self._neumann_bcs.extend(grouped.get(NeumannBC, []))

    def to_problem(
        self,
        form_compiler_options: dict[str, str] | None = None,
        jit_options: dict[str, str] | None = None,
    ) -> IncrSmallStrainProblem:
        self._apply_neumann_bcs()
        return IncrSmallStrainProblem(
            self._law_on_submeshs,
            self.incr_disp,
            self.stress,
            self.tangent,
            self.del_t,
            self.R_form,
            self._dirichlet_bcs,
            self.dR_form,
            form_compiler_options=form_compiler_options or {},
            jit_options=jit_options if jit_options is not None else {},
        )

    def _apply_neumann_bcs(self) -> None:
        if not self._neumann_bcs:
            return

        mesh = self.u.function_space.mesh
        mesh_tags = self._tag_neumann_boundaries(mesh)
        dA = ufl.Measure("ds", domain=mesh, subdomain_data=mesh_tags)
        V = self.u.function_space
        test_function = ufl.TestFunction(V)

        for bc in self._neumann_bcs:
            self.R_form -= bc.term(test_function, dA)

    def _tag_neumann_boundaries(self, mesh) -> MeshTags:
        entity_indices = [self._locate_entities(bc) for bc in self._neumann_bcs]
        entity_markers = [
            np.full_like(entities, bc.marker)
            for bc, entities in zip(self._neumann_bcs, entity_indices, strict=True)
        ]
        entity_indices = np.hstack(entity_indices).astype(np.int32)
        entity_markers = np.hstack(entity_markers).astype(np.int32)

        sorted_facets = np.argsort(entity_indices)
        entity_dim = mesh.topology.dim - 1
        return df.mesh.meshtags(
            mesh,
            entity_dim,
            entity_indices[sorted_facets],
            entity_markers[sorted_facets],
        )

    def _locate_entities(self, bc: NeumannBC) -> np.ndarray:
        V = self.u.function_space
        mesh = V.mesh
        entity_dim = mesh.topology.dim - 1
        return df.mesh.locate_entities(mesh, entity_dim, bc.boundary)
