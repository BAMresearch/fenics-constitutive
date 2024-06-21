from __future__ import annotations
import dolfinx as df
import numpy as np
from dolfinx.nls.petsc import NewtonSolver
from mises_plasticity_isotropic_hardening import VonMises3D
from mpi4py import MPI
from fenics_constitutive import Constraint, IncrSmallStrainProblem
from linear_elasticity_model import LinearElasticityModel
import matplotlib.pyplot as plt

mesh_update = True
co_rotation = True

def test_uniaxial_strain_3d():
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1, df.mesh.CellType.hexahedron)
    V = df.fem.VectorFunctionSpace(mesh, ("CG", 1))
    u = df.fem.Function(V)
    # matparam = {
    #     "p_ka": 175000,
    #     "p_mu": 80769,
    #     "p_y0": 1200,
    #     "p_y00": 2500,
    #     "p_w": 200,
    # }
    # law = VonMises3D(matparam)

    youngs_modulus = 42.0
    poissons_ratio = 0.3
    law = LinearElasticityModel(
        parameters={"E": youngs_modulus, "nu": poissons_ratio},
        constraint=Constraint.FULL,
    )

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

    # Define constants for boundary conditions
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
    displacement = []
    load = []
    invariant1 = []
    invariant2 = []
    invariant3 = []
    princ_stresses = []
    # V_stress = df.fem.VectorFunctionSpace(mesh, ("DG", 2), dim=6)
    # stress_tensor = df.fem.Function(V_stress)

    with df.io.XDMFFile(mesh.comm, "Mesh_.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        # for inc, disp in enumerate(disp_steps):
        #
        #     print("Stretch Increment:", inc)
        #     stretch_x.value[0] = disp
        #     problem.bcs = dirichlet_stretch
        #
        #     niter, converged = solver.solve(u)
        #     problem.update()
        #
        #     print(f"Converged: {converged} in {niter} iterations.")
        #     iterations = np.append(iterations, niter)
        #
        #     u.name = "Deformation"
        #     xdmf.write_function(u, float(inc))
        #
        #     stress_values = []
        #     stress_values.append(problem.stress_0.x.array.copy())
        #
        #     stress_values = stress_values[0]
        #     stress_element = stress_values[0:6]
        #
        #     stress_matrix = np.zeros((3, 3), dtype=np.float64)
        #
        #     stress_matrix[0, 0] = stress_element[0]
        #     stress_matrix[1, 1] = stress_element[1]
        #     stress_matrix[2, 2] = stress_element[2]
        #     stress_matrix[0, 1] = 1 / 2 ** 0.5 * (stress_element[3])
        #     stress_matrix[1, 2] = 1 / 2 ** 0.5 * (stress_element[4])
        #     stress_matrix[0, 2] = 1 / 2 ** 0.5 * (stress_element[5])
        #     stress_matrix[1, 0] = stress_element[1]
        #     stress_matrix[2, 1] = stress_element[2]
        #     stress_matrix[2, 0] = stress_element[2]
        #
        #     # trace_sigma = stress_element[0] + stress_element[1] + stress_element[2]
        #
        #     I1 = np.trace(stress_matrix)
        #     I2 = 0.5 * (I1 ** 2 - np.trace(stress_matrix @ stress_matrix))
        #     I3 = np.linalg.det(stress_matrix)
        #
        #     principal_stresses = np.linalg.eigvalsh(stress_matrix)
        #
        #     princ_stresses.append(principal_stresses)
        #
        #     invariant1.append(I1)
        #     invariant2.append(I2)
        #     invariant3.append(I3)
        #
        #     # invariant1.append(stress_element[0] + stress_element[1] + stress_element[2])
        #
        #     displacement.append(disp)
        #     load.append(stress_element)

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

            # niter, converged = solver.solve(u)
            # problem.update()


            if inc>0 :
                constants[1].value += (rotated_coords[1] - rotated_coords[0]) * 0.02
                constants[3].value += (rotated_coords[3] - rotated_coords[2]) * 0.02
                constants[5].value += (rotated_coords[5] - rotated_coords[4]) * 0.02
                constants[7].value += (rotated_coords[7] - rotated_coords[6]) * 0.02

            problem.bcs = dirichlet_rot

            niter, converged = solver.solve(u)
            problem.update()


            print(f"Converged: {converged} in {niter} iterations.")
            iterations = np.append(iterations, niter)

            stress_values = []
            stress_values.append(problem.stress_0.x.array.copy())
            stress_values = stress_values[0]
            stress_element = stress_values[0:6]

            stress_matrix = np.zeros((3, 3), dtype=np.float64)

            stress_matrix[0, 0] = stress_element[0]
            stress_matrix[1, 1] = stress_element[1]
            stress_matrix[2, 2] = stress_element[2]
            stress_matrix[0, 1] = 1 / 2 ** 0.5 * (stress_element[3])
            stress_matrix[1, 2] = 1 / 2 ** 0.5 * (stress_element[4])
            stress_matrix[0, 2] = 1 / 2 ** 0.5 * (stress_element[5])
            stress_matrix[1, 0] = stress_element[1]
            stress_matrix[2, 1] = stress_element[2]
            stress_matrix[2, 0] = stress_element[2]

            trace_sigma = stress_element[0]+stress_element[1]+stress_element[2]

            I1 = np.trace(stress_matrix)
            I2 = 0.5 * (I1 ** 2 - np.trace(stress_matrix @ stress_matrix))
            I3 = np.linalg.det(stress_matrix)

            principal_stresses = np.linalg.eigvalsh(stress_matrix)

            princ_stresses.append(principal_stresses)
            invariant1.append(I1)
            invariant2.append(I2)
            invariant3.append(I3)
            # print('1st Stress Invariant is ', stress_element[0]+stress_element[1]+stress_element[2])
            print(stress_element)
            displacement.append(angle)

            load.append(stress_element)

            u.name = "Deformation"
            xdmf.write_function(u, float(inc+21))

    displacement = np.array(displacement)
    load = np.array(load)

    ax = plt.subplots()[1]
    ax.plot(np.linspace(0, n_steps_rot+1, n_steps_rot+1), load[:,:], label=['xx','yy','zz', 'xy','yz','xz'])
    # ax.plot(np.linspace(0, 21, 21), invariant1, label='invariant 1')
    # ax.plot(np.linspace(0, 21, 21), invariant2, label='invariant 2')
    # ax.plot(np.linspace(0, 21, 21), invariant3, label='invariant 3')
    # ax.plot(np.linspace(0, 21, 21), princ_stresses, label=['principal stress 1','principal stress 2','principal stress 3'])
    ax.set_xlabel(r"$time$")
    ax.set_ylabel(r"$\sigma$")
    ax.legend()
    plt.show()


test_uniaxial_strain_3d()




# import dolfinx as df
# import numpy as np
# from dolfinx.nls.petsc import NewtonSolver
# from mises_plasticity_isotropic_hardening import VonMises3D
# from mpi4py import MPI
# from fenics_constitutive import Constraint, IncrSmallStrainProblem
# import matplotlib.pyplot as plt
# from linear_elasticity_model import LinearElasticityModel
#
# youngs_modulus = 42.0
# poissons_ratio = 0.3
#
#
# def corner_0_0_0(x):
#     return np.logical_and(
#         np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0)),
#         np.isclose(x[2], 0.0),
#     )
#
#
# def corner_0_1_0(x):
#     return np.logical_and(
#         np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 1.0)),
#         np.isclose(x[2], 0.0),
#     )
#
#
# def corner_0_0_1(x):
#     return np.logical_and(
#         np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0)),
#         np.isclose(x[2], 1.0),
#     )
#
#
# def corner_0_1_1(x):
#     return np.logical_and(
#         np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 1.0)),
#         np.isclose(x[2], 1.0),
#     )
#
#
# def corner_1_0_0(x):
#     return np.logical_and(
#         np.logical_and(np.isclose(x[0], 2.0), np.isclose(x[1], 0.0)),
#         np.isclose(x[2], 0.0),
#     )
#
#
# def corner_1_1_0(x):
#     return np.logical_and(
#         np.logical_and(np.isclose(x[0], 2.0), np.isclose(x[1], 1.0)),
#         np.isclose(x[2], 0.0),
#     )
#
#
# def corner_1_0_1(x):
#     return np.logical_and(
#         np.logical_and(np.isclose(x[0], 2.0), np.isclose(x[1], 0.0)),
#         np.isclose(x[2], 1.0),
#     )
#
#
# def corner_1_1_1(x):
#     return np.logical_and(
#         np.logical_and(np.isclose(x[0], 2.0), np.isclose(x[1], 1.0)),
#         np.isclose(x[2], 1.0),
#     )
#
#
# def test_uniaxial_strain_3d():
#     # Define the linear elasticity model
#     law = LinearElasticityModel(
#         parameters={"E": youngs_modulus, "nu": poissons_ratio},
#         constraint=Constraint.FULL,
#     )
#
#     def get_rotation_matrix(angle):
#         return np.array([
#             [np.cos(angle), -np.sin(angle), 0],
#             [np.sin(angle), np.cos(angle), 0],
#             [0, 0, 1]
#         ])
#
#     def apply_rotation(x, rotation_matrix):
#         return np.dot(rotation_matrix, x)
#
#     def update_boundary_conditions(angle, constants, corner_coords):
#         rotation_matrix = get_rotation_matrix(angle)
#         rotated_coords = [apply_rotation(coord, rotation_matrix) for coord in corner_coords]
#
#         for i, coord in enumerate(corner_coords):
#             rotated_coord = rotated_coords[i]
#             constants[i].value = rotated_coord-corner_coords[i]
#
#     # mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1, df.mesh.CellType.hexahedron)
#     mesh = df.mesh.create_box(
#         MPI.COMM_WORLD,
#         [np.array([0.0, 0.0, 0.0]), np.array([2, 1., 1.])],
#         [1, 1, 1],  # Number of cells in each direction
#         df.mesh.CellType.hexahedron
#     )
#     V = df.fem.VectorFunctionSpace(mesh, ("CG", 1))
#     u = df.fem.Function(V)
#
#     # Define constants for boundary conditions
#     constants = [df.fem.Constant(mesh, np.array([0.0, 0.0, 0.0])) for _ in range(8)]
#     # print(constants[5].value)
#
#     # Define coordinates for each corner
#     corner_coords = [
#         np.array([0, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([0, 1, 1]),
#         np.array([2, 0, 0]), np.array([2, 1, 0]), np.array([2, 0, 1]), np.array([2, 1, 1])
#     ]
#
#     corner_funcs = [corner_0_0_0, corner_0_1_0, corner_0_0_1, corner_0_1_1,
#                     corner_1_0_0,corner_1_1_0,corner_1_0_1,corner_1_1_1]
#
#     # Define boundary conditions using the constants
#     # drich
#     # for i in range(8):
#     #     dirichlet
#     dirichlet = [
#         df.fem.dirichletbc(constants[i], df.fem.locate_dofs_geometrical(V, corner_funcs[i]),V)
#         for i in range(8)
#     ]
#
#     problem = IncrSmallStrainProblem(law, u, [], q_degree=4, mesh_update=True, co_rotation=False)
#     solver = NewtonSolver(MPI.COMM_WORLD, problem)
#
#     n_steps = 10
#     total_angle = np.pi /2 # 90 degrees
#     angle_steps = np.linspace(0, total_angle, n_steps + 1)
#     iterations = np.array([], dtype=np.int32)
#     displacement = [0.0]
#     load = [0.0]
#
#
#     with df.io.XDMFFile(mesh.comm, "Mesh_2.xdmf", "w") as xdmf:
#         xdmf.write_mesh(mesh)
#
#         for inc, angle in enumerate(angle_steps):
#             print("Rotation Increment:", inc)
#
#             # Update constants with rotated coordinates
#             update_boundary_conditions(angle, constants, corner_coords)
#             problem.bcs = dirichlet
#             for i in range(8):
#                 print('ff',problem.bcs[i].g.value)
#
#             niter, converged = solver.solve(u)
#             problem.update()
#
#             print(f"Converged: {converged} in {niter} iterations.")
#             iterations = np.append(iterations, niter)
#
#             stress_values = []
#             stress_values.append(problem.stress_0.x.array.copy())
#             stress_values = stress_values[0]
#             stress_values = stress_values[::6][0]
#
#             displacement.append(angle)
#             load.append(stress_values)
#
#             u.name = "Deformation"
#             xdmf.write_function(u, float(inc))
#
#     displacement = np.array(displacement)
#     load = np.array(load)
#
#
# test_uniaxial_strain_3d()
