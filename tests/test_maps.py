from __future__ import annotations

import basix.ufl
from collections.abc import Callable

import dolfinx as df
from fenics_constitutive.maps import SubSpaceMap
import numpy as np
import pytest
import ufl
from mpi4py import MPI

from fenics_constitutive import build_subspace_map
from fenics_constitutive.maps import IdentityMap, SubSpaceMap

ElementBuilder = Callable[[df.mesh.Mesh], basix.ufl._ElementBase]

ELEMENT_BUILDERS = [
    lambda mesh: basix.ufl.quadrature_element(
        mesh.topology.cell_name(), value_shape=(1,), degree=1
    ),
    lambda mesh: basix.ufl.quadrature_element(
        mesh.topology.cell_name(), value_shape=(3,), degree=1
    ),
    lambda mesh: basix.ufl.quadrature_element(
        mesh.topology.cell_name(), value_shape=(3, 3), degree=1
    ),
]


@pytest.mark.mpi
def test_subspace_vector_map_vector_equals_tensor_map():
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 5, 7, 11)

    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local
    size_global = map_c.size_global
    cells_global = np.arange(0, size_global, dtype=np.int32)
    # cells = np.arange(0, num_cells, dtype=np.int32)

    Q = basix.ufl.quadrature_element(
        mesh.topology.cell_name(), value_shape=(1,), degree=2
    )
    QV = basix.ufl.quadrature_element(
        mesh.topology.cell_name(), value_shape=(3,), degree=2
    )
    QT = basix.ufl.quadrature_element(
        mesh.topology.cell_name(), value_shape=(3, 3), degree=2
    )

    Q_space = df.fem.functionspace(mesh, Q)
    QV_space = df.fem.functionspace(mesh, QV)
    QT_space = df.fem.functionspace(mesh, QT)

    rng = np.random.default_rng(42)
    if MPI.COMM_WORLD.rank == 0:
        cell_sample_global = rng.choice(cells_global, size_global // 2, replace=False)
    else:
        cell_sample_global = np.empty(size_global // 2, dtype=np.int32)
    MPI.COMM_WORLD.Bcast(cell_sample_global, root=0)

    cell_sample = map_c.global_to_local(cell_sample_global)
    cell_sample = cell_sample[cell_sample >= 0]

    Q_map, *_ = build_subspace_map(cell_sample, Q_space)
    QV_map, *_ = build_subspace_map(cell_sample, QV_space)
    QT_map, *_ = build_subspace_map(cell_sample, QT_space)

    assert isinstance(Q_map, SubSpaceMap)
    assert isinstance(QV_map, SubSpaceMap)
    assert isinstance(QT_map, SubSpaceMap)
    assert np.all(Q_map.parent == QV_map.parent)
    assert np.all(Q_map.child == QV_map.child)
    assert np.all(Q_map.parent == QT_map.parent)
    assert np.all(Q_map.child == QT_map.child)


@pytest.mark.mpi
@pytest.mark.parametrize("element_builder", ELEMENT_BUILDERS)
def test_subspace_map_evaluation(element_builder: ElementBuilder) -> None:
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 5, 7, 11, cell_type=df.mesh.CellType.hexahedron)

    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells_total = map_c.size_local + map_c.num_ghosts

    size_global = map_c.size_global
    cells_global = np.arange(0, size_global, dtype=np.int32)

    Q = element_builder(mesh)
    Q_space = df.fem.functionspace(mesh, Q)

    q = df.fem.Function(Q_space)
    q_test = q.copy()

    rng = np.random.default_rng(42)
    value_array = rng.random(q.x.array.shape)
    q.x.array[:] = value_array
    q.x.scatter_forward()
    value_array = q.x.array

    for _ in range(10):
        if MPI.COMM_WORLD.rank == 0:
            cell_sample_global = rng.choice(
                cells_global, size_global // 2, replace=False
            )
        else:
            cell_sample_global = np.empty(size_global // 2, dtype=np.int32)
        MPI.COMM_WORLD.Bcast(cell_sample_global, root=0)

        cell_sample = map_c.global_to_local(cell_sample_global)
        cell_sample = cell_sample[cell_sample >= 0]

        Q_sub_map, submesh, _ = build_subspace_map(cell_sample, Q_space)
        Q_sub = df.fem.functionspace(submesh, Q)
        q_sub = df.fem.Function(Q_sub)

        Q_sub_map.map_to_child(q, q_sub)
        Q_sub_map.map_to_parent(q_sub, q_test)

        q_view = value_array.reshape(num_cells_total, -1)
        q_test_view = q_test.x.array.reshape(num_cells_total, -1)
        assert np.all(q_view[cell_sample] == q_test_view[cell_sample])


@pytest.mark.parametrize("element_builder", ELEMENT_BUILDERS)
def test__identity_map_evaluation(element_builder: ElementBuilder) -> None:
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 5, 7, 11)

    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    Q = element_builder(mesh)
    Q_space = df.fem.functionspace(mesh, Q)

    q = df.fem.Function(Q_space)
    q_test = q.copy()

    rng = np.random.default_rng(42)
    value_array = rng.random(q.x.array.shape)
    q.x.array[:] = value_array

    identity_map, _, Q_sub = build_subspace_map(cells, Q_space)
    q_sub = df.fem.Function(Q_sub)

    identity_map.map_to_child(q, q_sub)
    identity_map.map_to_parent(q_sub, q_test)

    q_view = value_array.reshape(num_cells, -1)
    q_test_view = q_test.x.array.reshape(num_cells, -1)
    assert np.all(q_view == q_test_view)


if __name__ == "__main__":
    test_subspace_vector_map_vector_equals_tensor_map()
    for builder in ELEMENT_BUILDERS:
        test_subspace_map_evaluation(builder)
