from __future__ import annotations

from dataclasses import dataclass

import basix.ufl
import dolfinx as df

from fenics_constitutive.models.interfaces import StressStrainConstraint


@dataclass(frozen=True, slots=True)
class ElementSpaces:
    _stress_vector_element: basix.ufl._ElementBase
    _stress_tensor_element: basix.ufl._ElementBase
    _displacement_gradient_tensor_element: basix.ufl._ElementBase
    stress_vector_space: df.fem.FunctionSpace
    q_degree: int

    @staticmethod
    def create(
        mesh: df.mesh.Mesh, constraint: StressStrainConstraint, q_degree: int
    ) -> ElementSpaces:
        gdim = mesh.geometry.dim
        stress_vector_element = basix.ufl.quadrature_element(
            mesh.topology.cell_name(),
            value_shape=(constraint.stress_strain_dim,),
            degree=q_degree,
        )
        stress_tensor_element = basix.ufl.quadrature_element(
            mesh.topology.cell_name(),
            value_shape=(
                constraint.stress_strain_dim,
                constraint.stress_strain_dim,
            ),
            degree=q_degree,
        )
        displacement_gradient_tensor_element = basix.ufl.quadrature_element(
            mesh.topology.cell_name(), value_shape=(gdim, gdim), degree=q_degree
        )
        stress_vector_space = df.fem.functionspace(mesh, stress_vector_element)
        return ElementSpaces(
            stress_vector_element,
            stress_tensor_element,
            displacement_gradient_tensor_element,
            stress_vector_space,
            q_degree,
        )

    def displacement_gradient_tensor_space(
        self, mesh: df.mesh.Mesh
    ) -> df.fem.FunctionSpace:
        return df.fem.functionspace(mesh, self._displacement_gradient_tensor_element)

    def stress_tensor_space(self, mesh: df.mesh.Mesh) -> df.fem.FunctionSpace:
        return df.fem.functionspace(mesh, self._stress_tensor_element)


@dataclass(frozen=True, slots=True)
class GradientElementSpaces:
    _stress_vector_element: basix.ufl._ElementBase
    _local_quantity_element: basix.ufl._ElementBase
    _dsigma_deps_element: basix.ufl._ElementBase
    _dsigma_dnonlocal_element: basix.ufl._ElementBase
    _dlocal_deps_element: basix.ufl._ElementBase
    _dlocal_dnonlocal_element: basix.ufl._ElementBase
    _displacement_gradient_tensor_element: basix.ufl._ElementBase
    q_degree: int

    @staticmethod
    def create(
        mesh: df.mesh.Mesh, constraint: StressStrainConstraint, q_degree: int
    ) -> GradientElementSpaces:
        gdim = mesh.geometry.dim
        stress_vector_element = basix.ufl.quadrature_element(
            mesh.topology.cell_name(),
            value_shape=(constraint.stress_strain_dim,),
            degree=q_degree,
        )
        local_quantity_element = basix.ufl.quadrature_element(
            mesh.topology.cell_name(),
            value_shape=(1,),
            degree=q_degree,
        )
        dsigma_deps_element = basix.ufl.quadrature_element(
            mesh.topology.cell_name(),
            value_shape=(
                constraint.stress_strain_dim,
                constraint.stress_strain_dim,
            ),
            degree=q_degree,
        )
        dsigma_dnonlocal_element = basix.ufl.quadrature_element(
            mesh.topology.cell_name(),
            value_shape=(constraint.stress_strain_dim,),
            degree=q_degree,
        )
        dlocal_deps_element = basix.ufl.quadrature_element(
            mesh.topology.cell_name(),
            value_shape=(constraint.stress_strain_dim,),
            degree=q_degree,
        )
        dlocal_dnonlocal_element = basix.ufl.quadrature_element(
            mesh.topology.cell_name(),
            value_shape=(1,),
            degree=q_degree,
        )
        displacement_gradient_tensor_element = basix.ufl.quadrature_element(
            mesh.topology.cell_name(), value_shape=(gdim, gdim), degree=q_degree
        )
        # stress_vector_space = df.fem.functionspace(mesh, stress_vector_element)
        return GradientElementSpaces(
            stress_vector_element,
            local_quantity_element,
            dsigma_deps_element,
            dsigma_dnonlocal_element,
            dlocal_deps_element,
            dlocal_dnonlocal_element,
            displacement_gradient_tensor_element,
            q_degree,
        )

    def displacement_gradient_space(self, mesh: df.mesh.Mesh) -> df.fem.FunctionSpace:
        return df.fem.functionspace(mesh, self._displacement_gradient_tensor_element)

    def tangent_spaces(self, mesh: df.mesh.Mesh) -> list[df.fem.FunctionSpace]:
        return [
            df.fem.functionspace(mesh, self._dsigma_deps_element),
            df.fem.functionspace(mesh, self._dsigma_dnonlocal_element),
            df.fem.functionspace(mesh, self._dlocal_deps_element),
            df.fem.functionspace(mesh, self._dlocal_dnonlocal_element),
        ]

    def stress_vector_space(self, mesh: df.mesh.Mesh) -> df.fem.FunctionSpace:
        return df.fem.functionspace(mesh, self._stress_vector_element)
    def local_quantity_space(self, mesh: df.mesh.Mesh)-> df.fem.FunctionSpace:
        return df.fem.functionspace(mesh, self._local_quantity_element)

