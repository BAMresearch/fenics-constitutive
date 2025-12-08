from __future__ import annotations
import dolfinx as df
import numpy as np
import pytest
from dolfinx.nls.petsc import NewtonSolver
from fenics_constitutive.solver import CorotationalIncrSmallStrainProblem
from fenics_constitutive.models import LinearElasticityModel, StressStrainConstraint
from mpi4py import MPI

@pytest.mark.parametrize("experiment", ['Stretch_then_Rotate', 'Stretch_and_Rotate'])
def test_uniaxial_strain_3d(experiment):
    """
        Test objectivity under combined stretch and rotation in 3D.

        A unit cube is subjected to uniaxial stretch in the x-direction
        and rotated 90 degrees around the z-axis in two different ways:

        - "Stretch_then_Rotate":
            1) Apply pure stretch to a prescribed displacement.
            2) Afterwards, apply a rigid-body rotation.
        - "Stretch_and_Rotate":
            1) Stretch and rotate simultaneously during loading.

        For a linear elastic, objective formulation, the resulting
        stress state after a 90° rotation should:
            - have σ_xx = σ_zz (due to rotation of the principal directions),
            - have vanishing shear components (σ_xy, σ_yz, σ_xz ≈ 0).

        The test checks these conditions for both loading conditions.
        """
    # -------------------------------------------------------------------------
    # Mesh and function space
    # -------------------------------------------------------------------------
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1, df.mesh.CellType.hexahedron)
    V = df.fem.functionspace(mesh, ("CG", 1, (3,)))
    u = df.fem.Function(V)

    # -------------------------------------------------------------------------
    # Material model: linear elasticity in 3D
    # -------------------------------------------------------------------------
    youngs_modulus = 42.0
    poissons_ratio = 0.3
    law = LinearElasticityModel(
        parameters={"E": youngs_modulus, "nu": poissons_ratio},
        constraint=StressStrainConstraint.FULL,
    )

    # Define constant for stretch in x-direction
    stretch_x = df.fem.Constant(mesh, [0.0, 0.0, 0.0])

    # -------------------------------------------------------------------------
    # Corner indicator functions (8 vertices of the unit cube)
    # Each function returns True at exactly one corner.
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Dirichlet boundary conditions for stretching:
    # - All four left corners fixed at zero displacement
    # - All four right corners move with stretch_x in x-direction
    # -------------------------------------------------------------------------
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

    dirichlet_stretch = [
        fix_ux_left_000,
        fix_ux_left_010,
        fix_ux_left_001,
        fix_ux_left_011,
        fix_ux_left_100,
        fix_ux_left_110,
        fix_ux_left_101,
        fix_ux_left_111,
    ]

    # -------------------------------------------------------------------------
    # Dirichlet boundary conditions for rotation:
    # one Constant per corner, updated each increment
    # -------------------------------------------------------------------------
    constants = [df.fem.Constant(mesh, np.array([0.0, 0.0, 0.0])) for _ in range(8)]
    corner_funcs = [corner_0_0_0, corner_1_0_0, corner_0_1_0, corner_1_1_0,
                    corner_0_0_1, corner_1_0_1, corner_0_1_1, corner_1_1_1]

    dirichlet_rot = [
        df.fem.dirichletbc(constants[i], df.fem.locate_dofs_geometrical(V, corner_funcs[i]), V)
        for i in range(8)
    ]

    # -------------------------------------------------------------------------
    # Kinematics: rotation around z-axis
    # -------------------------------------------------------------------------

    def get_rotation_matrix(angle):
        """Return 3D rotation matrix for a rotation about the z-axis."""
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

    def apply_rotation(x, rotation_matrix):
        """Apply 3x3 rotation matrix to a 3D vector."""
        return np.dot(rotation_matrix, x)

    def update_boundary_conditions(angle, constants, corner_coords):
        """
        Update rotational Dirichlet BCs for a given incremental rotation angle.

        The rotation is applied around the z-axis, and the Dirichlet values
        are prescribed as the incremental displacement from the original corner
        positions, plus the current solution u at those corners. This mimics
        a rigid-body rotation superposed on the current deformation state.
        """
        rotation_matrix = get_rotation_matrix(angle)
        rotated_coords = [apply_rotation(coord, rotation_matrix) for coord in corner_coords]

        # Incremental displacement for each corner = (rotated - original) + current displacement
        for i, coord in enumerate(corner_coords):
            constants[i].value = rotated_coords[i] - corner_coords[i] + u.x.array.reshape(-1,3)[i]

        return rotated_coords

    # -------------------------------------------------------------------------
    # Problem and solver: use corotational small-strain solver
    # -------------------------------------------------------------------------
    problem = CorotationalIncrSmallStrainProblem(law, u, [], q_degree=4)
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

    # -------------------------------------------------------------------------
    # Main loading loop: first stretching (if requested), then rotation.
    # Results (deformation + stress) are written to XDMF for inspection.
    # -------------------------------------------------------------------------
    with (df.io.XDMFFile(mesh.comm, filename, "w") as xdmf):
        xdmf.write_mesh(mesh)

        # --- Phase 1: pure stretch in x-direction ---
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

        # --- Phase 2: rotation about the z-axis ---
        for inc, angle in enumerate(angle_steps):

           # Define coordinates for each corner
            corner_coords = mesh.geometry.x

            print("Rotation Increment:", inc)
            if inc == 0:
                increment_angle = 0
            else:
                increment_angle = angle_steps[1] - angle_steps[0]

            rotated_coords = update_boundary_conditions(increment_angle, constants, corner_coords)

            # For the "Stretch_and_Rotate" case, superpose a small additional stretch
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

    # -------------------------------------------------------------------------
    # Postprocessing: convert list of stresses to array and plot
    # -------------------------------------------------------------------------
    load = np.array(load)

    # -------------------------------------------------------------------------
    # Checks for objectivity after 90° rotation:
    # - σ_xx and σ_zz must coincide
    # - all shear components must vanish
    # -------------------------------------------------------------------------
    assert load[-1,0] - load[-1,2] <= 1e-8 # σ_xx == σ_zz
    assert load[-1,3] + load[-1,4] + load[-1,5] <= 1e-8 # σ_xy, σ_yz, σ_xz ≈ 0




