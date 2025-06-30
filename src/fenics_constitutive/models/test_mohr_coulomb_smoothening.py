from __future__ import annotations

import dolfinx as df
import numpy as np
from dolfinx.nls.petsc import NewtonSolver
from mises_plasticity_isotropic_hardening import VonMises3D
from mohr_coulomb_smoothed import mohr_coulomb_smoothed_3D
from mohr_coulomb_analytical_implementation import mohr_coulomb_smoothed_3D_analytical
from mpi4py import MPI
from fenics_constitutive import Constraint, IncrSmallStrainProblem
import matplotlib.pyplot as plt

def uniaxial_strain_3d():
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1, df.mesh.CellType.hexahedron)
    V = df.fem.VectorFunctionSpace(mesh, ("CG", 1))
    u = df.fem.Function(V)
    matparam = {
        "E": 6778,
        "nu": 0.3,
        "c": 3.45,
        "phi":30 * np.pi / 180 ,
        "psi": 30 * np.pi / 180,
        "theta_T": 26 * np.pi / 180 ,
        "a": 0.25 * 3.45 / np.tan(30),
    }

    # law = mohr_coulomb_smoothed_3D(matparam)
    law = mohr_coulomb_smoothed_3D_analytical(matparam)

    def left(x):
        return np.isclose(x[0], 0.0)

    def right(x):
        return np.isclose(x[0], 1.0)

    def upper_boundary(x):
        return np.isclose(x[1], 0.0)

    def side_boundary(x):
        return np.isclose(x[2], 0.0)
    #
    def corner_0_0_0(x):
        return np.logical_and(
            np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0)),
            np.isclose(x[2], 0.0),
        )
    #
    # def corner_0_1_0(x):
    #     return np.logical_and(
    #         np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 1.0)),
    #         np.isclose(x[2], 0.0),
    #     )
    #
    # def corner_0_0_1(x):
    #     return np.logical_and(
    #         np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0)),
    #         np.isclose(x[2], 1.0),
    #     )
    #
    # def corner_0_1_1(x):
    #     return np.logical_and(
    #         np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 1.0)),
    #         np.isclose(x[2], 1.0),
    #     )
    #
    # def corner_1_0_0(x):
    #     return np.logical_and(
    #         np.logical_and(np.isclose(x[0], 1.0), np.isclose(x[1], 0.0)),
    #         np.isclose(x[2], 0.0),
    #     )
    #
    # def corner_1_1_0(x):
    #     return np.logical_and(
    #         np.logical_and(np.isclose(x[0], 1.0), np.isclose(x[1], 1.0)),
    #         np.isclose(x[2], 0.0),
    #     )
    #
    # def corner_1_0_1(x):
    #     return np.logical_and(
    #         np.logical_and(np.isclose(x[0], 1.0), np.isclose(x[1], 0.0)),
    #         np.isclose(x[2], 1.0),
    #     )
    #
    # def corner_1_1_1(x):
    #     return np.logical_and(
    #         np.logical_and(np.isclose(x[0], 1.0), np.isclose(x[1], 1.0)),
    #         np.isclose(x[2], 1.0),
    #     )

    tdim = mesh.topology.dim
    fdim = tdim - 1

    left_facets = df.mesh.locate_entities_boundary(mesh, fdim, left)
    right_facets = df.mesh.locate_entities_boundary(mesh, fdim, right)
    upper_facets = df.mesh.locate_entities_boundary(mesh, fdim, upper_boundary)
    side_facets = df.mesh.locate_entities_boundary(mesh, fdim, side_boundary)
    # point_000_facet = df.mesh.locate_entities_boundary(mesh, fdim, corner_0_0_0)
    #
    # ### Dirichlet BCs
    zero_scalar = df.fem.Constant(mesh, 0.0)
    # scalar_x = df.fem.Constant(mesh, [0.0, 0.0, 0.0])
    scalar_x = df.fem.Constant(mesh, 0.015)

    fix_ux_left_000 = df.fem.dirichletbc(
        np.array([0.0, 0.0, 0.0]),
        df.fem.locate_dofs_geometrical(V, corner_0_0_0),
        V,
    )

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
    fix_uy = df.fem.dirichletbc(
        zero_scalar,
        df.fem.locate_dofs_topological(V.sub(1), fdim, upper_facets),
        V.sub(1),
    )
    fix_uz = df.fem.dirichletbc(
        zero_scalar,
        df.fem.locate_dofs_topological(V.sub(2), fdim, side_facets),
        V.sub(2),
    )

    dirichlet = [fix_ux_left, move_ux_right, fix_uy, fix_uz, fix_ux_left_000]
    # dirichlet = [fix_ux_left, move_ux_right]
    #


    #######################################################################

    # fix_ux_left_000 = df.fem.dirichletbc(
    #     np.array([0.0, 0.0, 0.0]),
    #     df.fem.locate_dofs_geometrical(V, corner_0_0_0),
    #     V,
    # )
    #
    # fix_ux_left_010 = df.fem.dirichletbc(
    #     np.array([0.0, 0.0, 0.0]),
    #     df.fem.locate_dofs_geometrical(V, corner_0_1_0),
    #     V,
    # )
    #
    # fix_ux_left_001 = df.fem.dirichletbc(
    #     np.array([0.0, 0.0, 0.0]),
    #     df.fem.locate_dofs_geometrical(V, corner_0_0_1),
    #     V,
    # )
    # fix_ux_left_011 = df.fem.dirichletbc(
    #         np.array([0.0, 0.0, 0.0]),
    #         df.fem.locate_dofs_geometrical(V, corner_0_1_1),
    #         V,
    #     )
    #
    # fix_ux_left_100 = df.fem.dirichletbc(
    #     scalar_x,
    #     df.fem.locate_dofs_geometrical(V, corner_1_0_0),
    #     V,
    # )
    #
    # fix_ux_left_110 = df.fem.dirichletbc(
    #     scalar_x,
    #     df.fem.locate_dofs_geometrical(V, corner_1_1_0),
    #     V,
    # )
    #
    # fix_ux_left_101 = df.fem.dirichletbc(
    #     scalar_x,
    #     df.fem.locate_dofs_geometrical(V, corner_1_0_1),
    #     V,
    # )
    # fix_ux_left_111 = df.fem.dirichletbc(
    #     scalar_x,
    #     df.fem.locate_dofs_geometrical(V, corner_1_1_1),
    #     V,
    # )

    # move_ux_right = df.fem.dirichletbc(
    #     scalar_x,
    #     df.fem.locate_dofs_topological(V.sub(0), fdim, right_facets),
    #     V.sub(0),
    # )
    #
    # dirichlet = [fix_ux_left_000,fix_ux_left_010,fix_ux_left_001,fix_ux_left_011,
    #              fix_ux_left_100,fix_ux_left_110,fix_ux_left_101,fix_ux_left_111,
    #              ]


    problem = IncrSmallStrainProblem(law, u, dirichlet, q_degree=2, mesh_update=False, co_rotation=False)

    solver = NewtonSolver(MPI.COMM_WORLD, problem)

    nTime = 5
    max_disp = 0.005
    load_steps = np.linspace(0, 1, num=nTime + 1)[1:]
    iterations = np.array([], dtype=np.int32)
    displacement = [0.0]
    load = [0.0]

    with df.io.XDMFFile(mesh.comm, "Mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        for inc, time in enumerate(load_steps):
            print("Load Increment:", inc)

            current_disp = time * max_disp
            scalar_x.value = (-current_disp)
            # scalar_x_2.value = (-1.0*current_disp)

            # problem.bcs = dirichlet


            niter, converged = solver.solve(u)
            problem.update()

            print(f"Converged: {converged} in {niter} iterations.")
            iterations = np.append(iterations, niter)

            stress_values = []
            stress_values.append(problem.stress_0.x.array.copy())
            print(np.shape(stress_values))
            stress_values = stress_values[0]
            stress_values = stress_values[::6][0]

            displacement.append(current_disp)
            load.append(stress_values)

            u.name = "Deformation"
            xdmf.write_function(u, float(inc))

    # with df.io.XDMFFile(mesh.comm, "Mesh_updated.xdmf", "w") as xdmf:
    #     xdmf.write_mesh(mesh)

    displacement = np.array(displacement)
    load = np.array(load)

    # Apply rotation to the mesh
    # coordinates = mesh.geometry.x
    # theta = np.pi / 2  # 90 degrees in radians
    # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
    #                             [np.sin(theta), np.cos(theta), 0],
    #                             [0, 0, 1]])
    #
    # # Rotate coordinates
    # new_coordinates = rotation_matrix @ coordinates.T
    # mesh.geometry.x[:] = new_coordinates.T

    # with df.io.XDMFFile(mesh.comm, "Mesh_deformed_rotated.xdmf", "w") as xdmf:
    #     xdmf.write_mesh(mesh)
    #     # niter, converged = solver.solve(u)
    #     # problem.update()
    #     # Optionally write function u or other results
    #     u.name = "Deformation_rotated"
    #     xdmf.write_function(u)

    # if the maximum stress exceeds the yield limit
    # tolerance = 1e-8
    # # assert np.max(load) - matparam["p_y00"] <= tolerance
    #
    # # if material behaves linearly under the elastic range with correct slope
    # indices = load + tolerance < matparam["p_y0"]
    # v = (3 * matparam["p_ka"] - 2 * matparam["p_mu"]) / (2 * (3 * matparam["p_ka"] + matparam["p_mu"]))
    # trace = displacement[indices][1] - 2 * v * displacement[indices][1]
    # dev = displacement[indices][1] - trace / 3
    # slope = (matparam["p_ka"] * trace + 2 * matparam["p_mu"] * dev) / displacement[indices][1]
    # # assert np.all(abs(np.ediff1d(load[indices]) / np.ediff1d(displacement[indices]) - slope) < 1e-7)

    ax = plt.subplots()[1]

    ax.plot(displacement, load, label="num")
    ax.set_xlabel(r"$\varepsilon_{xx}$")
    ax.set_ylabel(r"$\sigma_{xx}$")
    ax.legend()
    plt.savefig("figure.png", dpi=300, bbox_inches="tight")  # Save as PNG with high resolution
    plt.show()

def test_uniaxial_cyclic_strain_3d():
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    V = df.fem.VectorFunctionSpace(mesh, ("CG", 1))
    u = df.fem.Function(V)
    matparam = {
        "E": 6778,
        "nu": 0.3,
        "c": 3.45,
        "phi": 30 * np.pi / 180,
        "psi": 30 * np.pi / 180,
        "theta_T": 26 * np.pi / 180,
        "a": 0.25 * 3.45 / np.tan(30),
    }
    law = mohr_coulomb_smoothed_3D(matparam)

    def left(x):
        return np.isclose(x[0], 0.0)

    def right(x):
        return np.isclose(x[0], 1.0)

    def upper_boundary(x):
        return np.isclose(x[1], 0.0)

    def side_boundary(x):
        return np.isclose(x[2], 0.0)

    tdim = mesh.topology.dim
    fdim = tdim - 1

    left_facets = df.mesh.locate_entities_boundary(mesh, fdim, left)
    right_facets = df.mesh.locate_entities_boundary(mesh, fdim, right)
    upper_facets = df.mesh.locate_entities_boundary(mesh, fdim, upper_boundary)
    side_facets = df.mesh.locate_entities_boundary(mesh, fdim, side_boundary)
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
    fix_uy = df.fem.dirichletbc(
        zero_scalar,
        df.fem.locate_dofs_topological(V.sub(1), fdim, upper_facets),
        V.sub(1),
    )
    fix_uz = df.fem.dirichletbc(
        zero_scalar,
        df.fem.locate_dofs_topological(V.sub(2), fdim, side_facets),
        V.sub(2),
    )

    dirichlet = [fix_ux_left, move_ux_right, fix_uy, fix_uz]
    #
    problem = IncrSmallStrainProblem(law, u, dirichlet, q_degree=2)

    solver = NewtonSolver(MPI.COMM_WORLD, problem)

    nTime = 10
    max_disp = 0.005
    load_steps = np.linspace(np.pi, -np.pi, num=nTime + 1)
    iterations = np.array([], dtype=np.int32)
    displacement = [0.0]
    load = [0.0]
    # df.log.set_log_level(df.log.LogLevel.INFO)
    for inc, time in enumerate(load_steps):
        print("Load Increment:", inc)

        current_disp = np.sin(time) * max_disp
        scalar_x.value = (current_disp)

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
    # assert np.max(load) - matparam["p_y00"] <= tolerance
    # assert abs(np.min(load)) - matparam["p_y00"] <= tolerance

    # if material behaves linearly under the elastic range in 1/4 loading phase with correct slope
    load_interval_1 = load[:int(nTime / 4 + 2)]
    disp_interval_1 = displacement[:int(nTime / 4 + 2)]
    # indices = abs(load_interval_1) + tolerance < matparam["p_y0"]
    # v = (3 * matparam["p_ka"] - 2 * matparam["p_mu"]) / (2 * (3 * matparam["p_ka"] + matparam["p_mu"]))
    # trace = disp_interval_1[indices][1] - 2 * v * disp_interval_1[indices][1]
    # dev = disp_interval_1[indices][1] - trace / 3
    # slope = ((matparam["p_ka"] * trace + 2 * matparam["p_mu"] * dev) / disp_interval_1[indices][1])
    # assert np.all(abs(np.ediff1d(load_interval_1[indices][1:]) / np.ediff1d(disp_interval_1[indices][1:]) - slope) < 1e-7)

    # if material behaves linearly under the elastic range in 2/4 and 3/4 loading phase with correct slope
    # also consider if the elastic range has been stretched
    # load_interval_2 = load[int(nTime / 4 + 2):int(3*nTime / 4 + 1)]
    # disp_interval_2 = displacement[int(nTime / 4 + 2):int(3*nTime / 4 + 1)]
    # indices = abs(load_interval_2) + tolerance < max(np.max(load_interval_1), matparam["p_y0"])
    # assert np.all(abs(np.ediff1d(load_interval_2[indices]) / np.ediff1d(disp_interval_2[indices]) - slope) < 1e-7)

    # if material behaves linearly under the elastic range in 4/4 loading phase with correct slope
    # also consider if the elastic range has been stretched
    # load_interval_3 = load[int(3 * nTime / 4 + 1):]
    # disp_interval_3 = displacement[int(3 * nTime / 4 + 1):]
    # indices = abs(load_interval_3) + tolerance < max(np.max(load_interval_1),abs(np.min(load_interval_2)), matparam["p_y0"])
    # assert np.all(abs(np.ediff1d(load_interval_3[indices]) / np.ediff1d(disp_interval_3[indices]) - slope) < 1e-7)

    ax = plt.subplots()[1]
    ax.plot(displacement, load, label="num")
    ax.set_xlabel(r"$\varepsilon_{xx}$")
    ax.set_ylabel(r"$\sigma_{xx}$")
    ax.legend()
    plt.show()

uniaxial_strain_3d()
# test_uniaxial_cyclic_strain_3d()