from __future__ import annotations

import dolfinx as df
import numpy as np
import pytest
import ufl
from dolfinx.nls.petsc import NewtonSolver
from spring_kelvin_model import SpringKelvinModel
from spring_maxwell_model import SpringMaxwellModel
from mpi4py import MPI

from fenics_constitutive import Constraint, IncrSmallStrainProblem, IncrSmallStrainModel

youngs_modulus = 42.0
poissons_ratio = 0.2
visco_modulus = 10.0
relaxation_time = 10.0

@pytest.mark.parametrize("mat", [SpringKelvinModel, SpringMaxwellModel])
def test_relaxation_uniaxial_stress(mat: IncrSmallStrainModel):
    mesh = df.mesh.create_unit_interval(MPI.COMM_WORLD, 2)
    V = df.fem.FunctionSpace(mesh, ("CG", 1))
    u = df.fem.Function(V)
    law = mat(
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
    total_time = 20*relaxation_time
    while time[-1] < total_time:
        time.append(time[-1]+dt)
        niter, converged = solver.solve(u)
        problem.update()
        # print(f"time {time[-1]} Converged: {converged} in {niter} iterations.")

        # print(problem.stress_1.x.array)  # mandel stress at time t
        # print(u.x.array)
        disp.append(u.x.array[-1])
        stress.append(problem.stress_1.x.array[-1])
        strain.append(problem._history_1[0]['strain'].x.array[-1])
        viscostrain.append(problem._history_1[0]['strain_visco'].x.array[-1])

    # print(disp, stress, strain, viscostrain)
    # analytic solution
    if isinstance(law,SpringKelvinModel):
        #analytic solution for 1D Kelvin model
        stress_0_ana = youngs_modulus * displacement.value/1.
        stress_final_ana = youngs_modulus * visco_modulus / (youngs_modulus + visco_modulus) * displacement.value/1.
    elif isinstance(law, SpringMaxwellModel):
        #analytic solution for 1D Maxwell model
        stress_0_ana = (youngs_modulus + visco_modulus) * displacement.value/1.
        stress_final_ana = youngs_modulus * displacement.value/1.

    else:
        assert False, "Model not implemented"

    # print(stress_0_ana, stress_final_ana)
    assert abs(stress[0] - stress_0_ana) < 1e-8
    assert abs(stress[-1] - stress_final_ana) < 1e-8
    assert abs(strain[0] - displacement.value/1) < 1e-8

    # sanity checks
    assert np.sum(np.diff(strain)) < 1e-8
    assert abs(viscostrain[0] - 0) < 1e-8
    assert viscostrain[-1] > 0

@pytest.mark.parametrize("mat", [SpringKelvinModel, SpringMaxwellModel])
@pytest.mark.parametrize("case", [{'dim': 2, 'constraint': 'plane_stress'},
                                   {'dim': 2, 'constraint': 'plane_strain'},
                                   {'dim': 3}])
def test_relaxation(case: dict, mat: IncrSmallStrainModel):
    if case['dim'] == 2:
        mesh = df.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
        bc_vector = np.array([0.0, 0.0])

        if case['constraint'] == 'plane_stress':
            law = mat(
                parameters={"E0": youngs_modulus, "E1": visco_modulus, "tau": relaxation_time, "nu": poissons_ratio},
                constraint=Constraint.PLANE_STRESS,
            )
        elif case['constraint'] == 'plane_strain':
            law = mat(
                parameters={"E0": youngs_modulus, "E1": visco_modulus, "tau": relaxation_time, "nu": poissons_ratio},
                constraint=Constraint.PLANE_STRAIN,
            )

    elif case['dim'] == 3:
        mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
        bc_vector = np.array([0.0, 0.0, 0.0])

        law = mat(
            parameters={"E0": youngs_modulus, "E1": visco_modulus, "tau": relaxation_time, "nu": poissons_ratio},
            constraint=Constraint.FULL,
        )

    else:
        raise ValueError(f"Dimension {case['dim']} not supported")

    V = df.fem.VectorFunctionSpace(mesh, ("CG", 1))
    u = df.fem.Function(V)

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


    strain.append(problem._history_1[0]['strain'].x.array.max())
    viscostrain.append(problem._history_1[0]['strain_visco'].x.array.max())
    disp.append(u.x.array.max())
    stress.append(problem.stress_1.x.array.max())


    # set time step and solve until total time
    dt = 2
    problem._time = dt
    total_time = 20 * relaxation_time
    while time[-1] < total_time:
        time.append(time[-1] + dt)
        niter, converged = solver.solve(u)
        problem.update()
        print(f"time {time[-1]} Converged: {converged} in {niter} iterations.")

        # print(problem.stress_1.x.array)  # mandel stress at time t
        # print(u.x.array)
        disp.append(u.x.array.max())
        stress.append(problem.stress_1.x.array.max())
        strain.append(problem._history_1[0]['strain'].x.array.max())
        viscostrain.append(problem._history_1[0]['strain_visco'].x.array.max())

    print(disp[-1], stress[0], stress[-1], strain[0], viscostrain[0], viscostrain[-1])

    # analytic solution
    if isinstance(law,SpringKelvinModel):
        #analytic solution for 1D Kelvin model
        stress_0_ana = youngs_modulus * displacement/1.
        stress_final_ana = youngs_modulus * visco_modulus / (youngs_modulus + visco_modulus) * displacement/1.
    elif isinstance(law, SpringMaxwellModel):
        #analytic solution for 1D Maxwell model
        stress_0_ana = (youngs_modulus + visco_modulus) * displacement/1.
        stress_final_ana = youngs_modulus * displacement/1.

    else:
        assert False, "Model not implemented"

    print('ana', stress_0_ana, stress_final_ana)

    assert abs(stress[0] - stress_0_ana) < 1e-8
    assert abs(stress[-1] - stress_final_ana) < 1e-8
    assert abs(strain[0] - displacement/1) < 1e-8

    # sanity checks
    assert np.sum(np.diff(strain)) < 1e-8
    assert abs(viscostrain[0] - 0) < 1e-8
    assert viscostrain[-1] > 0

def test_kelvin_vs_maxwell():

    mesh = df.mesh.create_unit_interval(MPI.COMM_WORLD, 2)
    V = df.fem.FunctionSpace(mesh, ("CG", 1))
    u = df.fem.Function(V)

    #Kelvin material
    law_K = SpringKelvinModel(
        parameters={"E0": youngs_modulus, "E1": visco_modulus, "tau": relaxation_time, "nu": poissons_ratio},
        constraint=Constraint.UNIAXIAL_STRESS,
    )
    # transfer Kelvin parameters to Maxwell (see Technische Mechanik 4)
    E0_M = (youngs_modulus * visco_modulus) / (youngs_modulus + visco_modulus)
    E1_M = youngs_modulus ** 2 / (youngs_modulus + visco_modulus)
    tau_M = (youngs_modulus / (youngs_modulus + visco_modulus)) ** 2 * relaxation_time

    #Maxwell material
    law_M = SpringMaxwellModel(
        parameters={"E0": E0_M, "E1": E1_M, "tau": tau_M, "nu": poissons_ratio},
        constraint=Constraint.UNIAXIAL_STRESS,
    )

    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    def right_boundary(x):
        return np.isclose(x[0], 1.0)

    displacement = df.fem.Constant(mesh, 0.001)
    dofs_left = df.fem.locate_dofs_geometrical(V, left_boundary)
    dofs_right = df.fem.locate_dofs_geometrical(V, right_boundary)
    bc_left = df.fem.dirichletbc(df.fem.Constant(mesh, 0.0), dofs_left, V)
    bc_right = df.fem.dirichletbc(displacement, dofs_right, V)

    # solve Kelvin problem without linear step
    problems = [IncrSmallStrainProblem(
        law_K,
        u,
        [bc_left, bc_right],
        4,
    ), IncrSmallStrainProblem(
        law_M,
        u,
        [bc_left, bc_right],
        4,
    )]

    stress_p, strain_p = [], []

    # solve both problems
    for prob_i in problems:
        solver = NewtonSolver(MPI.COMM_WORLD, prob_i)

        time = [0]
        stress = []
        strain = []

        dt = 0.001
        prob_i._time = dt
        total_time = 10*dt #1*relaxation_time
        while time[-1] < total_time:
            time.append(time[-1]+dt)
            niter, converged = solver.solve(u)
            prob_i.update()
            #print(f"time {time[-1]} Converged: {converged} in {niter} iterations.")

            stress.append(prob_i.stress_1.x.array[-1])
            strain.append(prob_i._history_1[0]['strain'].x.array[-1])

        stress_p.append(stress)
        strain_p.append(strain)

    #print('Kelvin', stress_p[0], strain_p[0])
    #print('Maxwell', stress_p[1], strain_p[1])

    # print(np.linalg.norm(np.array(stress_p[0])-np.array(stress_p[1])))
    # accuracy depends on time step size!
    assert abs(np.linalg.norm(np.array(stress_p[0])-np.array(stress_p[1]))) < 1e-3



if __name__ == "__main__":

    # test_relaxation_uniaxial_stress(SpringKelvinModel)
    test_relaxation_uniaxial_stress(SpringMaxwellModel)

    # test_relaxation({'dim': 3}, SpringMaxwellModel)

    # test_kelvin_vs_maxwell()