from __future__ import annotations

import dolfinx as df
import numpy as np
from dolfinx.nls.petsc import NewtonSolver
from mises_plasticity_isotropic_hardening import VonMises3D
from mpi4py import MPI

from fenics_constitutive import IncrSmallStrainProblem


def test_uniaxial_strain_3d():
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    V = df.fem.VectorFunctionSpace(mesh, ("CG", 1))
    u = df.fem.Function(V)
    matparam = {
        "p_ka": 175000,
        "p_mu": 80769,
        "p_y0": 1200,
        "p_y00": 2500,
        "p_w": 200,
    }
    law = VonMises3D(matparam)

    def left(x):
        return np.isclose(x[0], 0.0)

    def right(x):
        return np.isclose(x[0], 1.0)

    tdim = mesh.topology.dim
    fdim = tdim - 1

    left_facets = df.mesh.locate_entities_boundary(mesh, fdim, left)
    right_facets = df.mesh.locate_entities_boundary(mesh, fdim, right)
    #
    # ### Dirichlet BCs
    zero_scalar = df.fem.Constant(mesh, 0.0)
    scalar_x = df.fem.Constant(mesh, 0.015)
    fix_ux_left = df.fem.dirichletbc(
        zero_scalar,
        df.fem.locate_dofs_topological(V.sub(0), fdim, left_facets),
        V.sub(0),
    )
    move_ux_right = df.fem.dirichletbc(
        scalar_x,
        df.fem.locate_dofs_topological(V.sub(0), fdim, right_facets),
        V.sub(0),
    )

    dirichlet = [fix_ux_left, move_ux_right]
    problem = IncrSmallStrainProblem(law, u, dirichlet, q_degree=2)

    solver = NewtonSolver(MPI.COMM_WORLD, problem)

    nTime = 100
    max_disp = 0.05
    load_steps = np.linspace(0, 1, num=nTime + 1)[1:]
    iterations = np.array([], dtype=np.int32)
    displacement = [0.0]
    load = [0.0]

    for inc, time in enumerate(load_steps):
        print("Load Increment:", inc)

        current_disp = time * max_disp
        scalar_x.value = current_disp

        niter, converged = solver.solve(u)
        problem.update()

        print(f"Converged: {converged} in {niter} iterations.")
        iterations = np.append(iterations, niter)

        stress_values = []
        stress_values.append(problem.stress_0.x.array.copy())
        stress_values = stress_values[0]
        stress_values = stress_values[::6][0]

        displacement.append(current_disp)
        load.append(stress_values)

    displacement = np.array(displacement)
    load = np.array(load)

    # if the maximum stress exceeds the yield limit
    tolerance = 1e-8
    assert np.max(load) - matparam["p_y00"] <= tolerance

    # if material behaves linearly under the elastic range with correct slope
    indices = load + tolerance < matparam["p_y0"]
    v = (3 * matparam["p_ka"] - 2 * matparam["p_mu"]) / (
        2 * (3 * matparam["p_ka"] + matparam["p_mu"])
    )
    trace = displacement[indices][1] - 2 * v * displacement[indices][1]
    dev = displacement[indices][1] - trace / 3
    slope = (matparam["p_ka"] * trace + 2 * matparam["p_mu"] * dev) / displacement[
        indices
    ][1]
    assert np.all(
        abs(np.ediff1d(load[indices]) / np.ediff1d(displacement[indices]) - slope)
        < 1e-7
    )


def test_uniaxial_cyclic_strain_3d():
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    V = df.fem.VectorFunctionSpace(mesh, ("CG", 1))
    u = df.fem.Function(V)
    matparam = {
        "p_ka": 175000,
        "p_mu": 80769,
        "p_y0": 1200,
        "p_y00": 2500,
        "p_w": 200,
    }
    law = VonMises3D(matparam)

    def left(x):
        return np.isclose(x[0], 0.0)

    def right(x):
        return np.isclose(x[0], 1.0)

    tdim = mesh.topology.dim
    fdim = tdim - 1

    left_facets = df.mesh.locate_entities_boundary(mesh, fdim, left)
    right_facets = df.mesh.locate_entities_boundary(mesh, fdim, right)
    #
    # ### Dirichlet BCs
    zero_scalar = df.fem.Constant(mesh, 0.0)
    scalar_x = df.fem.Constant(mesh, 0.015)
    fix_ux_left = df.fem.dirichletbc(
        zero_scalar,
        df.fem.locate_dofs_topological(V.sub(0), fdim, left_facets),
        V.sub(0),
    )
    move_ux_right = df.fem.dirichletbc(
        scalar_x,
        df.fem.locate_dofs_topological(V.sub(0), fdim, right_facets),
        V.sub(0),
    )

    dirichlet = [fix_ux_left, move_ux_right]
    problem = IncrSmallStrainProblem(law, u, dirichlet, q_degree=2)

    solver = NewtonSolver(MPI.COMM_WORLD, problem)

    nTime = 100
    max_disp = 0.05
    load_steps = np.linspace(np.pi, -np.pi, num=nTime + 1)
    iterations = np.array([], dtype=np.int32)
    displacement = [0.0]
    load = [0.0]

    for inc, time in enumerate(load_steps):
        print("Load Increment:", inc)

        current_disp = np.sin(time) * max_disp
        scalar_x.value = current_disp

        niter, converged = solver.solve(u)
        problem.update()

        print(f"Converged: {converged} in {niter} iterations.")
        iterations = np.append(iterations, niter)

        stress_values = []
        stress_values.append(problem.stress_0.x.array.copy())
        stress_values = stress_values[0]
        stress_values = stress_values[::6][0]

        displacement.append(current_disp)
        load.append(stress_values)

    displacement = np.array(displacement)
    load = np.array(load)

    # if the maximum and minimum stress exceeds the yield limit
    tolerance = 1e-8
    assert np.max(load) - matparam["p_y00"] <= tolerance
    assert abs(np.min(load)) - matparam["p_y00"] <= tolerance

    # if material behaves linearly under the elastic range in 1/4 loading phase with correct slope
    load_interval_1 = load[: int(nTime / 4 + 2)]
    disp_interval_1 = displacement[: int(nTime / 4 + 2)]
    indices = abs(load_interval_1) + tolerance < matparam["p_y0"]
    v = (3 * matparam["p_ka"] - 2 * matparam["p_mu"]) / (
        2 * (3 * matparam["p_ka"] + matparam["p_mu"])
    )
    trace = disp_interval_1[indices][1] - 2 * v * disp_interval_1[indices][1]
    dev = disp_interval_1[indices][1] - trace / 3
    slope = (matparam["p_ka"] * trace + 2 * matparam["p_mu"] * dev) / disp_interval_1[
        indices
    ][1]
    assert np.all(
        abs(
            np.ediff1d(load_interval_1[indices][1:])
            / np.ediff1d(disp_interval_1[indices][1:])
            - slope
        )
        < 1e-7
    )

    # if material behaves linearly under the elastic range in 2/4 and 3/4 loading phase with correct slope
    # also consider if the elastic range has been stretched
    load_interval_2 = load[int(nTime / 4 + 2) : int(3 * nTime / 4 + 1)]
    disp_interval_2 = displacement[int(nTime / 4 + 2) : int(3 * nTime / 4 + 1)]
    indices = abs(load_interval_2) + tolerance < max(
        np.max(load_interval_1), matparam["p_y0"]
    )
    assert np.all(
        abs(
            np.ediff1d(load_interval_2[indices]) / np.ediff1d(disp_interval_2[indices])
            - slope
        )
        < 1e-7
    )

    # if material behaves linearly under the elastic range in 4/4 loading phase with correct slope
    # also consider if the elastic range has been stretched
    load_interval_3 = load[int(3 * nTime / 4 + 1) :]
    disp_interval_3 = displacement[int(3 * nTime / 4 + 1) :]
    indices = abs(load_interval_3) + tolerance < max(
        np.max(load_interval_1), abs(np.min(load_interval_2)), matparam["p_y0"]
    )
    assert np.all(
        abs(
            np.ediff1d(load_interval_3[indices]) / np.ediff1d(disp_interval_3[indices])
            - slope
        )
        < 1e-7
    )


if __name__ == "__main__":
    test_uniaxial_strain_3d()
