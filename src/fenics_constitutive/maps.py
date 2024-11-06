from __future__ import annotations

from dataclasses import dataclass

import dolfinx as df
import numpy as np

from functools import reduce
import operator

__all__ = ["SubSpaceMap", "build_subspace_map"]


@dataclass
class SubSpaceMap:
    """
    Map between a subspace and a parent space.

    Args:
        parent: The dof indices of the parent space.
        child: The dof indices of the child space.
        sub_mesh: The mesh of the subspace.
        parent_mesh: The mesh of the parent space.

    """

    parent: np.ndarray
    child: np.ndarray
    sub_mesh: df.mesh.Mesh
    parent_mesh: df.mesh.Mesh

    @df.common.timed("constitutive: map_to_parent_mesh")
    def map_to_parent(self, sub: df.fem.Function, parent: df.fem.Function) -> None:
        """
        Map the values from the subspace to the parent space.

        Args:
            sub: The function in the subspace.
            parent: The function in the parent space.
        """
        assert sub.function_space.mesh == self.sub_mesh, "Subspace mesh does not match"
        assert (
            parent.function_space.mesh == self.parent_mesh
        ), "Parent mesh does not match"
        assert sub.ufl_shape == parent.ufl_shape, "Shapes do not match"

        size = reduce(operator.mul, sub.ufl_shape, 1)

        parent_array = parent.x.array.reshape(-1, size)
        sub_array = sub.x.array.reshape(-1, size)
        parent_array[self.parent] = sub_array[self.child]
        parent.x.scatter_forward()

    @df.common.timed("constitutive: map_to_child_mesh")
    def map_to_child(self, parent: df.fem.Function, sub: df.fem.Function) -> None:
        """
        Map the values from the parent space to the subspace.

        Args:
            parent: The function in the parent space.
            sub: The function in the subspace.
        """
        assert sub.function_space.mesh == self.sub_mesh, "Subspace mesh does not match"
        assert (
            parent.function_space.mesh == self.parent_mesh
        ), "Parent mesh does not match"
        assert sub.ufl_shape == parent.ufl_shape, "Shapes do not match"

        size = reduce(operator.mul, sub.ufl_shape, 1)

        parent_array = parent.x.array.reshape(-1, size)
        sub_array = sub.x.array.reshape(-1, size)
        sub_array[self.child] = parent_array[self.parent]
        sub.x.scatter_forward()


@df.common.timed("constitutive: build_subspace_map")
def build_subspace_map(
    cells: np.ndarray, V: df.fem.FunctionSpace, return_subspace=False
) -> (
    tuple[SubSpaceMap, df.mesh.Mesh]
    | tuple[SubSpaceMap, df.mesh.Mesh, df.fem.FunctionSpace]
):
    """
    Build a map between a subspace and a parent space. This currently needs
    to build a functionspace which can optionally be returned.

    Args:
        cells: The cells of the subspace.
        V: The parent function space.
        return_subspace: Return the subspace function space.

    Returns:
        The subspace map, the submesh and optionally the subspace function space.
    """
    mesh = V.mesh
    submesh, cell_map, _, _ = df.mesh.create_submesh(mesh, mesh.topology.dim, cells)
    fe = V.ufl_element()
    V_sub = df.fem.functionspace(submesh, fe)

    submesh = V_sub.mesh
    view_parent = []
    view_child = []

    map_c = submesh.topology.index_map(mesh.topology.dim)
    num_sub_cells = map_c.size_local + map_c.num_ghosts
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

    del V_sub
    return map, submesh
