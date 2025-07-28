from __future__ import annotations

import basix
import dolfinx as df
import numpy as np
import pytest
from mpi4py import MPI

from fenics_constitutive import build_subspace_map


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

    Q_map, _ = build_subspace_map(cell_sample, Q_space)
    QV_map, _ = build_subspace_map(cell_sample, QV_space)
    QT_map, _ = build_subspace_map(cell_sample, QT_space)

    assert np.all(Q_map.parent == QV_map.parent)
    assert np.all(Q_map.child == QV_map.child)
    assert np.all(Q_map.parent == QT_map.parent)
    assert np.all(Q_map.child == QT_map.child)


@pytest.mark.mpi
def test_map_evaluation():
    mesh = df.mesh.create_unit_cube(
        MPI.COMM_WORLD, 5, 4, 1, cell_type=df.mesh.CellType.hexahedron
    )

    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells_total = map_c.size_local + map_c.num_ghosts

    size_global = map_c.size_global
    cells_global = np.arange(0, size_global, dtype=np.int32)

    Q = basix.ufl.quadrature_element(
        mesh.topology.cell_name(), value_shape=(1,), degree=1
    )
    QV = basix.ufl.quadrature_element(
        mesh.topology.cell_name(), value_shape=(3,), degree=1
    )
    QT = basix.ufl.quadrature_element(
        mesh.topology.cell_name(), value_shape=(3, 3), degree=1
    )

    Q_space = df.fem.functionspace(mesh, Q)
    QV_space = df.fem.functionspace(mesh, QV)
    QT_space = df.fem.functionspace(mesh, QT)

    q = df.fem.Function(Q_space)
    q_test = q.copy()
    qv = df.fem.Function(QV_space)
    qv_test = qv.copy()
    qt = df.fem.Function(QT_space)
    qt_test = qt.copy()

    rng = np.random.default_rng(42)
    scalar_array = rng.random(q.x.array.shape)

    q.x.array[:] = scalar_array
    q.x.scatter_forward()
    scalar_array = q.x.array

    vector_array = rng.random(qv.x.array.shape)
    qv.x.array[:] = vector_array
    qv.x.scatter_forward()
    vector_array = qv.x.array

    tensor_array = rng.random(qt.x.array.shape)
    qt.x.array[:] = tensor_array
    qt.x.scatter_forward()
    tensor_array = qt.x.array

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

        Q_sub_map, submesh = build_subspace_map(
            cell_sample, Q_space, return_subspace=False
        )

        Q_sub = df.fem.functionspace(submesh, Q)
        QV_sub = df.fem.functionspace(submesh, QV)
        QT_sub = df.fem.functionspace(submesh, QT)

        q_sub = df.fem.Function(Q_sub)
        qv_sub = df.fem.Function(QV_sub)
        qt_sub = df.fem.Function(QT_sub)

        Q_sub_map.map_to_child(q, q_sub)
        Q_sub_map.map_to_child(qv, qv_sub)
        Q_sub_map.map_to_child(qt, qt_sub)

        Q_sub_map.map_to_parent(q_sub, q_test)
        Q_sub_map.map_to_parent(qv_sub, qv_test)
        Q_sub_map.map_to_parent(qt_sub, qt_test)

        q_view = scalar_array.reshape(num_cells_total, -1)
        q_test_view = q_test.x.array.reshape(num_cells_total, -1)
        assert np.all(q_view[cell_sample] == q_test_view[cell_sample])

        qv_view = vector_array.reshape(num_cells_total, -1)
        qv_test_view = qv_test.x.array.reshape(num_cells_total, -1)
        assert np.all(qv_view[cell_sample] == qv_test_view[cell_sample])

        qt_view = tensor_array.reshape(num_cells_total, -1)
        qt_test_view = qt_test.x.array.reshape(num_cells_total, -1)
        assert np.all(qt_view[cell_sample] == qt_test_view[cell_sample])


if __name__ == "__main__":
    test_subspace_vector_map_vector_equals_tensor_map()
    test_map_evaluation()
