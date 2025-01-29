from __future__ import annotations
import dolfinx as df
import numpy as np
import pytest
from dolfinx.nls.petsc import NewtonSolver
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples','linear_elasticity')))
from linear_elasticity_model import LinearElasticityModel
from mpi4py import MPI
from fenics_constitutive import Constraint, IncrSmallStrainProblem, strain_from_grad_u
import matplotlib.pyplot as plt

mesh_update = True
co_rotation = True

@pytest.mark.parametrize("experiment", ['Stretch_then_Rotate', 'Stretch_and_Rotate'])
def test_uniaxial_strain_3d(experiment):
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1, df.mesh.CellType.hexahedron)
    V = df.fem.VectorFunctionSpace(mesh, ("CG", 1))
    u = df.fem.Function(V)

    youngs_modulus = 42.0
    poissons_ratio = 0.3
    law = LinearElasticityModel(
        parameters={"E": youngs_modulus, "nu": poissons_ratio},
        constraint=Constraint.FULL,
    )

    # Define constant for stretch boundary condition
    stretch_x = df.fem.Constant(mesh, [0.0, 0.0, 0.0])

    def corner_0_0_0(x):
        return np.logical_and(
            np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0)),
            np.isclose(x[2], 0.0),
        )

    def corner_0_1_0(x):
        return np.logical_and(
            np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 1.0)),
            np.isclose(x[2], 0.0),
        )

    def corner_0_0_1(x):
        return np.logical_and(
            np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0)),
            np.isclose(x[2], 1.0),
        )

    def corner_0_1_1(x):
        return np.logical_and(
            np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 1.0)),
            np.isclose(x[2], 1.0),
        )

    #
    def corner_1_0_0(x):
        return np.logical_and(
            np.logical_and(np.isclose(x[0], 1.0), np.isclose(x[1], 0.0)),
            np.isclose(x[2], 0.0),
        )

    def corner_1_1_0(x):
        return np.logical_and(
            np.logical_and(np.isclose(x[0], 1.0), np.isclose(x[1], 1.0)),
            np.isclose(x[2], 0.0),
        )

    def corner_1_0_1(x):
        return np.logical_and(
            np.logical_and(np.isclose(x[0], 1.0), np.isclose(x[1], 0.0)),
            np.isclose(x[2], 1.0),
        )

    def corner_1_1_1(x):
        return np.logical_and(
            np.logical_and(np.isclose(x[0], 1.0), np.isclose(x[1], 1.0)),
            np.isclose(x[2], 1.0),
        )

    fix_ux_left_000 = df.fem.dirichletbc(
        np.array([0.0, 0.0, 0.0]),
        df.fem.locate_dofs_geometrical(V, corner_0_0_0),
        V,
    )

    fix_ux_left_010 = df.fem.dirichletbc(
        np.array([0.0, 0.0, 0.0]),
        df.fem.locate_dofs_geometrical(V, corner_0_1_0),
        V,
    )

    fix_ux_left_001 = df.fem.dirichletbc(
        np.array([0.0, 0.0, 0.0]),
        df.fem.locate_dofs_geometrical(V, corner_0_0_1),
        V,
    )
    fix_ux_left_011 = df.fem.dirichletbc(
        np.array([0.0, 0.0, 0.0]),
        df.fem.locate_dofs_geometrical(V, corner_0_1_1),
        V,
    )

    fix_ux_left_100 = df.fem.dirichletbc(
        stretch_x,
        df.fem.locate_dofs_geometrical(V, corner_1_0_0),
        V,
    )

    fix_ux_left_110 = df.fem.dirichletbc(
        stretch_x,
        df.fem.locate_dofs_geometrical(V, corner_1_1_0),
        V,
    )

    fix_ux_left_101 = df.fem.dirichletbc(
        stretch_x,
        df.fem.locate_dofs_geometrical(V, corner_1_0_1),
        V,
    )
    fix_ux_left_111 = df.fem.dirichletbc(
        stretch_x,
        df.fem.locate_dofs_geometrical(V, corner_1_1_1),
        V,
    )

    dirichlet_stretch = [fix_ux_left_000, fix_ux_left_010, fix_ux_left_001, fix_ux_left_011,
                 fix_ux_left_100, fix_ux_left_110, fix_ux_left_101, fix_ux_left_111,
                 ]

    # Define constants for rotational boundary conditions
    constants = [df.fem.Constant(mesh, np.array([0.0, 0.0, 0.0])) for _ in range(8)]
    corner_funcs = [corner_0_0_0, corner_1_0_0, corner_0_1_0, corner_1_1_0,
                    corner_0_0_1, corner_1_0_1, corner_0_1_1, corner_1_1_1]

    dirichlet_rot = [
        df.fem.dirichletbc(constants[i], df.fem.locate_dofs_geometrical(V, corner_funcs[i]), V)
        for i in range(8)
    ]

    def get_rotation_matrix(angle):
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

    def apply_rotation(x, rotation_matrix):
        return np.dot(rotation_matrix, x)

    def update_boundary_conditions(angle, constants, corner_coords):
        rotation_matrix = get_rotation_matrix(angle)
        rotated_coords = [apply_rotation(coord, rotation_matrix) for coord in corner_coords]

        for i, coord in enumerate(corner_coords):
            constants[i].value = rotated_coords[i]-corner_coords[i] + u.x.array.reshape(-1,3)[i]

        return rotated_coords

    problem = IncrSmallStrainProblem(law, u, [], q_degree=4, mesh_update=mesh_update, co_rotation=co_rotation)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)

    n_steps_stretch = 20
    n_steps_rot = 20
    total_angle = np.pi/2 # 90 degrees
    total_disp = 0.5
    angle_steps = np.linspace(0, total_angle, n_steps_rot + 1)
    disp_steps = np.linspace(0, total_disp, n_steps_stretch + 1)
    iterations = []
    load = []
    filename=experiment
    filename += ".xdmf"
    with (df.io.XDMFFile(mesh.comm, filename, "w") as xdmf):
        xdmf.write_mesh(mesh)
        for inc, disp in enumerate(disp_steps):

            if experiment=='Stretch_then_Rotate':

                print("Stretch Increment:", inc)
                stretch_x.value[0] = disp
                problem.bcs = dirichlet_stretch

                niter, converged = solver.solve(u)
                problem.update()

                print(f"Converged: {converged} in {niter} iterations.")
                iterations = np.append(iterations, niter)

                u.name = "Deformation"
                xdmf.write_function(u, float(inc))

                stress_values = []
                stress_values.append(problem.stress_0.x.array.copy())

                stress_values = stress_values[0]
                stress_element = stress_values[0:6]

                load.append(stress_element)

        for inc, angle in enumerate(angle_steps):

           # Define coordinates for each corner
            if mesh_update == True:
                corner_coords = mesh.geometry.x
            elif mesh_update == False:
                corner_coords = mesh.geometry.x + u.x.array.reshape(-1, 3)

            print("Rotation Increment:", inc)
            if inc == 0:
                increment_angle = 0
            else:
                increment_angle = angle_steps[1] - angle_steps[0]

            rotated_coords = update_boundary_conditions(increment_angle, constants, corner_coords)
            if experiment == 'Stretch_and_Rotate':
                if inc >0:
                   constants[1].value += (rotated_coords[1] - rotated_coords[0]) * 0.02
                   constants[3].value += (rotated_coords[3] - rotated_coords[2]) * 0.02
                   constants[5].value += (rotated_coords[5] - rotated_coords[4]) * 0.02
                   constants[7].value += (rotated_coords[7] - rotated_coords[6]) * 0.02

            problem.bcs = dirichlet_rot

            niter, converged = solver.solve(u)
            problem.update()

            print(f"Converged: {converged} in {niter} iterations.")
            iterations = np.append(iterations, niter)

            u.name = "Deformation"
            xdmf.write_function(u, float(inc + 21))

            stress_values = []
            stress_values.append(problem.stress_0.x.array.copy())
            stress_values = stress_values[0]
            stress_element = stress_values[0:6]

            load.append(stress_element)

    load = np.array(load)

    ax = plt.subplots()[1]
    if experiment=='Stretch_then_Rotate':
        ax.plot(np.linspace(0, n_steps_rot+n_steps_stretch+2, n_steps_rot+n_steps_stretch+2), load[:,:], label=['xx','yy','zz', 'xy','yz','xz'])
    elif experiment=='Stretch_and_Rotate':
        ax.plot(np.linspace(0, n_steps_rot+1, n_steps_rot+1), load[:,:], label=['xx','yy','zz', 'xy','yz','xz'])
    ax.set_xlabel(r"$time$")
    ax.set_ylabel(r"$\sigma$")
    ax.legend()
    plt.show()

    assert load[-1,0] - load[-1,2] <= 1e-8 # stress xx and zz should be equal after 90 degree rotation
    assert load[-1,3] + load[-1,4] + load[-1,5] <= 1e-8 # stress xy, yz, xz should be zero after 90 degree rotation


