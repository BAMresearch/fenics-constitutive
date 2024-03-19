import dolfinx as df
import numpy as np
from functools import reduce
from dataclasses import dataclass

__all__ = ["SubSpaceMap", "_build_view", "build_subspace_map"]


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
        # print(sub.ufl_element().value_size(), "sub.ufl_element().value_size()")
        size = sub.ufl_element().value_size()  # reduce(lambda x, y: x * y, sub.ufl_shape) if not isinstance(sub.ufl_shape, int) else sub.ufl_shape

        parent_array = parent.x.array.reshape(-1, size)
        sub_array = sub.x.array.reshape(-1, size)
        parent_array[self.parent] = sub_array[self.child]

    def map_to_child(self, parent: df.fem.Function, sub: df.fem.Function) -> None:
        assert sub.function_space.mesh == self.sub_mesh, "Subspace mesh does not match"
        assert (
            parent.function_space.mesh == self.parent_mesh
        ), "Parent mesh does not match"
        assert sub.ufl_shape == parent.ufl_shape, "Shapes do not match"

        # size = reduce(lambda x, y: x * y, sub.ufl_shape) if not isinstance(sub.ufl_shape, int) else sub.ufl_shape
        size = sub.ufl_element().value_size()  # reduce(lambda x, y: x * y, sub.ufl_shape) if not isinstance(sub.ufl_shape, int) else sub.ufl_shape

        parent_array = parent.x.array.reshape(-1, size)
        sub_array = sub.x.array.reshape(-1, size)
        parent_array[self.parent] = sub_array[self.child]


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


def _build_view(
    cells: np.ndarray, V: df.fem.FunctionSpace
) -> tuple[SubSpaceMap, df.fem.FunctionSpace]:
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
        return (
            SubSpaceMap(
                parent=np.hstack(view_parent),
                child=np.hstack(view_child),
                sub_mesh=submesh,
                parent_mesh=V.mesh,
            ),
            V_sub,
        )
    else:
        # it may be that a process does not own any of the cells in the submesh
        return (
            SubSpaceMap(
                parent=np.array([], dtype=np.int32),
                child=np.array([], dtype=np.int32),
                sub_mesh=submesh,
                parent_mesh=V.mesh,
            ),
            V_sub,
        )
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            V_sub,
        )
    # if view_child:
    #     return (
    #         np.hstack(view_parent),
    #         np.hstack(view_child),
    #         V_sub,
    #     )
    # else:
    #     # it may be that a process does not own any of the cells in the submesh
    #     return (
    #         np.array([], dtype=np.int32),
    #         np.array([], dtype=np.int32),
    #         V_sub,
    #     )
