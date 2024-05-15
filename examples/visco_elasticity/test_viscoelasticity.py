from __future__ import annotations

import dolfinx as df
import numpy as np
import pytest
import ufl
from dolfinx.nls.petsc import NewtonSolver
from spring_kelvin_model import SpringKelvinModel
from mpi4py import MPI

from fenics_constitutive import Constraint, IncrSmallStrainProblem

youngs_modulus = 42.0
poissons_ratio = 0.2
visco_modulus = 10.0
relaxation_time = 10.0


def test_relaxation_uniaxial_stress():
    mesh = df.mesh.create_unit_interval(MPI.COMM_WORLD, 2)
    V = df.fem.FunctionSpace(mesh, ("CG", 1))
    u = df.fem.Function(V)
    law = SpringKelvinModel(
        parameters={"E0": youngs_modulus, "E1": visco_modulus, "tau": relaxation_time, "nu": poissons_ratio},
        constraint=Constraint.UNIAXIAL_STRESS,
    )

    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    def right_boundary(x):
        return np.isclose(x[0], 1.0)

    displacement = df.fem.Constant(mesh, 0.01)
    dofs_left = df.fem.locate_dofs_geometrical(V, left_boundary)
    dofs_right = df.fem.locate_dofs_geometrical(V, right_boundary)
    bc_left = df.fem.dirichletbc(df.fem.Constant(mesh, 0.0), dofs_left, V)
    bc_right = df.fem.dirichletbc(displacement, dofs_right, V)

    problem = IncrSmallStrainProblem(
        law,
        u,
        [bc_left, bc_right],
        1,
    )

    solver = NewtonSolver(MPI.COMM_WORLD, problem)

    time = [0]
    disp = []
    stress = []
    strain = []
    viscostrain = []

    # elastic first step
    problem._time = 0
    solver.solve(u)
    problem.update()

    # store values last element/point
    disp.append(u.x.array[-1])
    stress.append(problem.stress_1.x.array[-1])
    strain.append(problem._history_1[0]['strain'].x.array[-1])
    viscostrain.append(problem._history_1[0]['strain_visco'].x.array[-1])

    # set time step and solve until total time
    dt = 2
    problem._time = dt
    total_time = 10*relaxation_time
    while time[-1] < total_time:
        time.append(time[-1]+dt)
        niter, converged = solver.solve(u)
        problem.update()
        print(f"time {time[-1]} Converged: {converged} in {niter} iterations.")

        # print(problem.stress_1.x.array)  # mandel stress at time t
        # print(u.x.array)
        disp.append(u.x.array[-1])
        stress.append(problem.stress_1.x.array[-1])
        strain.append(problem._history_1[0]['strain'].x.array[-1])
        viscostrain.append(problem._history_1[0]['strain_visco'].x.array[-1])

    # print(disp, stress)
    # print(strain, viscostrain)
    #analytic solution for 1D Kelvin model
    stress_0_ana = youngs_modulus * displacement.value/1.
    stress_final_ana = youngs_modulus * visco_modulus / (youngs_modulus + visco_modulus) * displacement.value/1.

    assert abs(stress[0] - stress_0_ana) < 1e-8
    assert abs(stress[-1] - stress_final_ana) < 1e-8
    assert abs(strain[0] - displacement.value/1) < 1e-8

    # sanity checks
    assert np.sum(np.diff(strain)) < 1e-8
    assert abs(viscostrain[0] - 0) < 1e-8
    assert viscostrain[-1] > 0

@pytest.mark.parametrize(constraint = ['plane_stress', 'plane_strain', 'full'])
@pytest.mark.parametrize(dim = [2,3])
def test_relaxation(constraint: str, dim: int):
    if dim == 2:
        mesh = df.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
        bc_vector = np.array([0.0, 0.0])
    elif dim == 3:
        mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
        bc_vector = np.array([0.0, 0.0, 0.0])
    else:
        raise ValueError(f"Dimension {dim} not supported")

    V = df.fem.VectorFunctionSpace(mesh, ("CG", 1))
    u = df.fem.Function(V)
    if constraint == 'plane_stress':
        law = SpringKelvinModel(
            parameters={"E0": youngs_modulus, "E1": visco_modulus, "tau": relaxation_time, "nu": poissons_ratio},
            constraint=Constraint.UNIAXIAL_STRESS,
        )
    elif constraint == 'plane_strain':
        law = SpringKelvinModel(
            parameters={"E0": youngs_modulus, "E1": visco_modulus, "tau": relaxation_time, "nu": poissons_ratio},
            constraint=Constraint.PLANE_STRAIN,
        )
    elif constraint == 'full':
        law = SpringKelvinModel(
            parameters={"E0": youngs_modulus, "E1": visco_modulus, "tau": relaxation_time, "nu": poissons_ratio},
            constraint=Constraint.FULL,
        )
    else:
        raise ValueError(f"Constraint {constraint} not supported")

    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    def right_boundary(x):
        return np.isclose(x[0], 1.0)

    displacement = 0.01
    dofs_left = df.fem.locate_dofs_geometrical(V, left_boundary)
    dofs_right = df.fem.locate_dofs_geometrical(V, right_boundary)
    bc_left = df.fem.dirichletbc(bc_vector, dofs_left, V)
    # displacement in x direction
    disp_vector = bc_vector.copy()
    disp_vector[0] = displacement
    bc_right = df.fem.dirichletbc(disp_vector, dofs_right, V)

    problem = IncrSmallStrainProblem(
        law,
        u,
        [bc_left, bc_right],
        1,
    )

    solver = NewtonSolver(MPI.COMM_WORLD, problem)

    time = [0]
    disp = []
    stress = []
    strain = []
    viscostrain = []

    # elastic first step
    problem._time = 0
    solver.solve(u)
    problem.update()
    print(u.x.array)
    print(problem.stress_1.x.array)


    strain.append(problem._history_1[0]['strain'].x.array[0])
    viscostrain.append(problem._history_1[0]['strain_visco'].x.array[0])
    disp.append(u.x.array.max())
    stress.append(problem.stress_1.x.array.max())


    # set time step and solve until total time
    dt = 2
    problem._time = dt
    total_time = 10 * relaxation_time
    while time[-1] < total_time:
        time.append(time[-1] + dt)
        niter, converged = solver.solve(u)
        problem.update()
        print(f"time {time} Converged: {converged} in {niter} iterations.")

        # print(problem.stress_1.x.array)  # mandel stress at time t
        # print(u.x.array)
        disp.append(u.x.array.max())
        stress.append(problem.stress_1.x.array.max())
        strain.append(problem._history_1[0]['strain'].x.array[0])
        viscostrain.append(problem._history_1[0]['strain_visco'].x.array[0])

    print(disp, stress)
    print(strain)
    # analytic solution for 1D Kelvin model
    stress_0_ana = youngs_modulus * displacement/1.
    stress_final_ana = youngs_modulus * visco_modulus / (youngs_modulus + visco_modulus) * displacement/1.
    print('ana', stress_0_ana, stress_final_ana)

    # # sanity check if out of plane stress is NOT zero
    # assert (
    #     np.linalg.norm(
    #         problem.stress_0.x.array.reshape(-1, law.constraint.stress_strain_dim())[
    #             :, 2
    #         ]
    #     )
    #     > 1e-2
    # )


if __name__ == "__main__":
    # test_relaxation_uniaxial_stress()

    test_relaxation('full',3)