from __future__ import annotations

from dataclasses import dataclass

import dolfinx as df
import numpy as np

__all__ = ["SubSpaceMap", "build_subspace_map"]


@dataclass
class SubSpaceMap:
    parent: np.ndarray
    child: np.ndarray
    sub_mesh: df.mesh.Mesh
    parent_mesh: df.mesh.Mesh

    def map_to_parent(self, sub: df.fem.Function, parent: df.fem.Function) -> None:
        assert sub.function_space.mesh == self.sub_mesh, "Subspace mesh does not match"
        assert (
            parent.function_space.mesh == self.parent_mesh
        ), "Parent mesh does not match"
        assert sub.ufl_shape == parent.ufl_shape, "Shapes do not match"

        size = sub.ufl_element().value_size()

        parent_array = parent.x.array.reshape(-1, size)
        sub_array = sub.x.array.reshape(-1, size)
        parent_array[self.parent] = sub_array[self.child]
        parent.x.scatter_forward()

    def map_to_child(self, parent: df.fem.Function, sub: df.fem.Function) -> None:
        assert sub.function_space.mesh == self.sub_mesh, "Subspace mesh does not match"
        assert (
            parent.function_space.mesh == self.parent_mesh
        ), "Parent mesh does not match"
        assert sub.ufl_shape == parent.ufl_shape, "Shapes do not match"

        size = sub.ufl_element().value_size()

        parent_array = parent.x.array.reshape(-1, size)
        sub_array = sub.x.array.reshape(-1, size)
        sub_array[self.child] = parent_array[self.parent]
        sub.x.scatter_forward()


def build_subspace_map(
    cells: np.ndarray, V: df.fem.FunctionSpace, return_subspace=False
) -> (
    tuple[SubSpaceMap, df.mesh.Mesh]
    | tuple[SubSpaceMap, df.mesh.Mesh, df.fem.FunctionSpace]
):
    mesh = V.mesh
    submesh, cell_map, _, _ = df.mesh.create_submesh(mesh, mesh.topology.dim, cells)
    fe = V.ufl_element()
    V_sub = df.fem.FunctionSpace(submesh, fe)

    submesh = V_sub.mesh
    view_parent = []
    view_child = []

    num_sub_cells = submesh.topology.index_map(submesh.topology.dim).size_local
    for cell in range(num_sub_cells):
        view_child.append(V_sub.dofmap.cell_dofs(cell))
        view_parent.append(V.dofmap.cell_dofs(cell_map[cell]))

    if view_child:
        map = SubSpaceMap(
            parent=np.hstack(view_parent),
            child=np.hstack(view_child),
            sub_mesh=submesh,
            parent_mesh=V.mesh,
        )
    else:
        map = SubSpaceMap(
            parent=np.array([], dtype=np.int32),
            child=np.array([], dtype=np.int32),
            sub_mesh=submesh,
            parent_mesh=V.mesh,
        )
    if return_subspace:
        return map, submesh, V_sub
    else:
        del V_sub
        return map, submesh
