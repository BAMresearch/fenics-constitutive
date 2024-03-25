from __future__ import annotations

import dolfinx as df
import numpy as np
import ufl
from mpi4py import MPI

from fenics_constitutive import build_subspace_map


def test_subspace_vector_map_vector_equals_tensor_map():
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 5, 7, 11)

    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    Q = ufl.FiniteElement("Quadrature", mesh.ufl_cell(), 2, quad_scheme="default")
    QV = ufl.VectorElement(
        "Quadrature", mesh.ufl_cell(), 2, quad_scheme="default", dim=3
    )
    QT = ufl.TensorElement(
        "Quadrature", mesh.ufl_cell(), 2, quad_scheme="default", shape=(3, 3)
    )

    Q_space = df.fem.FunctionSpace(mesh, Q)
    QV_space = df.fem.FunctionSpace(mesh, QV)
    QT_space = df.fem.FunctionSpace(mesh, QT)

    Q_map, _ = build_subspace_map(cells, Q_space)
    QV_map, _ = build_subspace_map(cells, QV_space)
    QT_map, _ = build_subspace_map(cells, QT_space)

    assert np.all(Q_map.parent == QV_map.parent)
    assert np.all(Q_map.child == QV_map.child)
    assert np.all(Q_map.parent == QT_map.parent)
    assert np.all(Q_map.child == QT_map.child)

    for _ in range(10):
        cell_sample = np.random.permutation(cells)[: num_cells // 2]

        Q_map, _ = build_subspace_map(cell_sample, Q_space)
        QV_map, _ = build_subspace_map(cell_sample, QV_space)
        QT_map, _ = build_subspace_map(cell_sample, QT_space)

        assert np.all(Q_map.parent == QV_map.parent)
        assert np.all(Q_map.child == QV_map.child)
        assert np.all(Q_map.parent == QT_map.parent)
        assert np.all(Q_map.child == QT_map.child)


def test_map_evaluation():
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 5, 7, 11)

    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    Q = ufl.FiniteElement("Quadrature", mesh.ufl_cell(), 2, quad_scheme="default")
    QV = ufl.VectorElement(
        "Quadrature", mesh.ufl_cell(), 2, quad_scheme="default", dim=3
    )
    QT = ufl.TensorElement(
        "Quadrature", mesh.ufl_cell(), 2, quad_scheme="default", shape=(3, 3)
    )

    Q_space = df.fem.FunctionSpace(mesh, Q)
    QV_space = df.fem.FunctionSpace(mesh, QV)
    QT_space = df.fem.FunctionSpace(mesh, QT)

    q = df.fem.Function(Q_space)
    q_test = q.copy()
    qv = df.fem.Function(QV_space)
    qv_test = qv.copy()
    qt = df.fem.Function(QT_space)
    qt_test = qt.copy()

    scalar_array = np.random.random(q.x.array.shape)
    q.x.array[:] = scalar_array
    vector_array = np.random.random(qv.x.array.shape)
    qv.x.array[:] = vector_array
    tensor_array = np.random.random(qt.x.array.shape)
    qt.x.array[:] = tensor_array

    for _ in range(10):
        cell_sample = np.random.permutation(cells)[: num_cells // 2]

        Q_sub_map, submesh = build_subspace_map(
            cell_sample, Q_space, return_subspace=False
        )
        Q_sub = df.fem.FunctionSpace(submesh, Q)
        QV_sub = df.fem.FunctionSpace(submesh, QV)
        QT_sub = df.fem.FunctionSpace(submesh, QT)

        q_sub = df.fem.Function(Q_sub)
        qv_sub = df.fem.Function(QV_sub)
        qt_sub = df.fem.Function(QT_sub)

        Q_sub_map.map_to_child(q, q_sub)
        Q_sub_map.map_to_child(qv, qv_sub)
        Q_sub_map.map_to_child(qt, qt_sub)

        Q_sub_map.map_to_parent(q_sub, q_test)
        Q_sub_map.map_to_parent(qv_sub, qv_test)
        Q_sub_map.map_to_parent(qt_sub, qt_test)

        q_view = scalar_array.reshape(num_cells, -1)
        q_test_view = q_test.x.array.reshape(num_cells, -1)
        assert np.all(q_view[cell_sample] == q_test_view[cell_sample])

        qv_view = vector_array.reshape(num_cells, -1)
        qv_test_view = qv_test.x.array.reshape(num_cells, -1)
        assert np.all(qv_view[cell_sample] == qv_test_view[cell_sample])

        qt_view = tensor_array.reshape(num_cells, -1)
        qt_test_view = qt_test.x.array.reshape(num_cells, -1)
        assert np.all(qt_view[cell_sample] == qt_test_view[cell_sample])
