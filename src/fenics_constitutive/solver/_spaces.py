from __future__ import annotations

from dataclasses import dataclass

import basix.ufl
import dolfinx as df

from fenics_constitutive.interfaces import StressStrainConstraint


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
