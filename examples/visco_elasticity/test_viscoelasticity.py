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
    '''stress relaxation under uniaxial tension test for 1D, displacement controlled'''

    mesh = df.mesh.create_unit_interval(MPI.COMM_WORLD, 2)
    V = df.fem.FunctionSpace(mesh, ("CG", 1))
    u = df.fem.Function(V)
    law = mat(
        parameters={"E0": youngs_modulus, "E1": visco_modulus, "tau": relaxation_time},
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
@pytest.mark.parametrize("dim", [2, 3])
def test_relaxation(dim: int, mat: IncrSmallStrainModel):
    '''stress relaxation under uniaxial tension test for 2D and 3D, displacement controlled, symmetric boundaries'''

    if dim == 2:
        mesh = df.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
        # only plane_stress otherwise pure uniaxial bc not possible
        law = mat(
            parameters={"E0": youngs_modulus, "E1": visco_modulus, "tau": relaxation_time, "nu": poissons_ratio},
            constraint=Constraint.PLANE_STRESS,
        )

    elif dim == 3:
        mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)

        law = mat(
            parameters={"E0": youngs_modulus, "E1": visco_modulus, "tau": relaxation_time, "nu": poissons_ratio},
            constraint=Constraint.FULL,
        )

        def z_boundary(x):
            return np.isclose(x[2], 0.0)

    else:
        raise ValueError(f"Dimension {case['dim']} not supported")

    V = df.fem.VectorFunctionSpace(mesh, ("CG", 1))
    u = df.fem.Function(V)

    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    def right_boundary(x):
        return np.isclose(x[0], 1.0)

    def y_boundary(x):
        return np.isclose(x[1], 0.0)

    displacement = 0.01

    # boundaries
    fdim = mesh.topology.dim - 1
    left_f = df.mesh.locate_entities_boundary(mesh, fdim, left_boundary)
    right_f = df.mesh.locate_entities_boundary(mesh, fdim, right_boundary)
    bc_y_f = df.mesh.locate_entities_boundary(mesh, fdim, y_boundary)

    # displacement in x direction
    move_ux_right = df.fem.dirichletbc(
        df.fem.Constant(mesh, displacement),
        df.fem.locate_dofs_topological(V.sub(0), fdim, right_f),
        V.sub(0),
    )

    # symmetric boundaries for modelling uniaxial tension
    zero_scalar = df.fem.Constant(mesh, 0.0)
    fix_ux = df.fem.dirichletbc(
        zero_scalar,
        df.fem.locate_dofs_topological(V.sub(0), fdim, left_f),
        V.sub(0),
    )
    fix_uy = df.fem.dirichletbc(
        zero_scalar,
        df.fem.locate_dofs_topological(V.sub(1), fdim, bc_y_f),
        V.sub(1),
    )
    dirc_bcs = [fix_ux, fix_uy, move_ux_right]

    if dim == 3:
        bc_z_f = df.mesh.locate_entities_boundary(mesh, fdim, z_boundary)
        fix_uz = df.fem.dirichletbc(
            zero_scalar,
            df.fem.locate_dofs_topological(V.sub(2), fdim, bc_z_f),
            V.sub(2),
        )
        dirc_bcs = [fix_ux, fix_uy, fix_uz, move_ux_right]

    # problem and solve
    problem = IncrSmallStrainProblem(
        law,
        u,
        dirc_bcs,
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
    """1D test with uniaxial stress, compare Kelvin and Maxwell model."""

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


@pytest.mark.parametrize("mat", [SpringKelvinModel, SpringMaxwellModel])
@pytest.mark.parametrize("case", [{'dim': 2, 'constraint': 'plane_stress'},
                                   {'dim': 3}])
def test_creep(case: dict, mat: IncrSmallStrainModel):
    '''creep under uniaxial tension test for 2D and 3D, stress controlled, symmetric boundaries'''

    f_max = 0.1
    if case['dim'] == 2:
        mesh = df.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
        load = (f_max, 0.0)

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
        load = (f_max, 0.0, 0.0)

        law = mat(
            parameters={"E0": youngs_modulus, "E1": visco_modulus, "tau": relaxation_time, "nu": poissons_ratio},
            constraint=Constraint.FULL,
        )

        def z_boundary(x):
            return np.isclose(x[2], 0.0)

    else:
        raise ValueError(f"Dimension {case['dim']} not supported")

    V = df.fem.VectorFunctionSpace(mesh, ("CG", 1))
    u = df.fem.Function(V)

    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    def right_boundary(x):
        return np.isclose(x[0], 1.0)

    def y_boundary(x):
        return np.isclose(x[1], 0.0)


    # boundaries
    fdim = mesh.topology.dim - 1
    left_f = df.mesh.locate_entities_boundary(mesh, fdim, left_boundary)
    bc_y_f = df.mesh.locate_entities_boundary(mesh, fdim, y_boundary)

    # symmetric boundaries for modelling uniaxial tension
    zero_scalar = df.fem.Constant(mesh, 0.0)
    fix_ux = df.fem.dirichletbc(
        zero_scalar,
        df.fem.locate_dofs_topological(V.sub(0), fdim, left_f),
        V.sub(0),
    )
    fix_uy = df.fem.dirichletbc(
        zero_scalar,
        df.fem.locate_dofs_topological(V.sub(1), fdim, bc_y_f),
        V.sub(1),
    )
    dirc_bcs = [fix_ux, fix_uy]

    if case['dim'] == 3:
        bc_z_f = df.mesh.locate_entities_boundary(mesh, fdim, z_boundary)
        fix_uz = df.fem.dirichletbc(
            zero_scalar,
            df.fem.locate_dofs_topological(V.sub(2), fdim, bc_z_f),
            V.sub(2),
        )
        dirc_bcs = [fix_ux, fix_uy, fix_uz]

    # loading in x
    neumann_tag = 15
    neumann_boundary = {"right": (neumann_tag, right_boundary)}
    facet_tags, _ = create_meshtags(mesh, mesh.topology.dim - 1, neumann_boundary)
    dA = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    neumann_data = df.fem.Constant(mesh, load)

    # problem and solve
    problem = IncrSmallStrainProblem(
        law,
        u,
        dirc_bcs,
        1,
    )
    # apply load
    test_function = ufl.TestFunction(V)
    fext = ufl.inner(neumann_data, test_function) * dA(neumann_tag)
    problem.R_form -= fext


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

        disp.append(u.x.array.max())
        stress.append(problem.stress_1.x.array.max())
        strain.append(problem._history_1[0]['strain'].x.array.max())
        viscostrain.append(problem._history_1[0]['strain_visco'].x.array.max())

    print(disp[-1], stress[0], stress[-1], strain[0], viscostrain[0], viscostrain[-1])

    # analytic solution
    if isinstance(law,SpringKelvinModel):
        #analytic solution for 1D Kelvin model
        strain_0_ana = (f_max/ 1) / youngs_modulus
        strain_final_ana = (f_max/ 1) / youngs_modulus + (f_max/ 1) / visco_modulus
    elif isinstance(law, SpringMaxwellModel):
        #analytic solution for 1D Maxwell model
        strain_0_ana = (f_max/ 1) / (youngs_modulus + visco_modulus)
        strain_final_ana = (f_max/ 1) / youngs_modulus

    else:
        assert False, "Model not implemented"

    print('ana', strain_0_ana, strain_final_ana)

    assert abs(strain[0] - strain_0_ana) < 1e-8
    assert abs(strain[-1] - strain_final_ana) < 1e-8
    assert abs(stress[0] - f_max/1) < 1e-8

    # sanity checks
    assert np.sum(np.diff(stress)) < 1e-8
    assert abs(viscostrain[0] - 0) < 1e-8
    assert viscostrain[-1] > 0

def create_meshtags(
        domain: df.mesh.Mesh, entity_dim: int, markers: dict[str, tuple[int, Callable]]
) -> tuple[df.mesh.MeshTagsMetaClass, dict[str, int]]:
    """Creates meshtags for the given markers.
    This code is part of the FEniCSx tutorial
    by Jørgen S. Dokken.
    See https://jsdokken.com/dolfinx-tutorial/chapter3/robin_neumann_dirichlet.html?highlight=sorted_facets#implementation # noqa: E501
    Args:
        domain: The computational domain.
        entity_dim: Dimension of the entities to mark.
        markers: The definition of subdomains or boundaries where each key is a string
          and each value is a tuple of an integer and a marker function.
    """
    tdim = domain.topology.dim
    assert entity_dim in (tdim, tdim - 1)

    entity_indices, entity_markers = [], []
    edim = entity_dim
    marked = {}
    for key, (marker, locator) in markers.items():
        entities = df.mesh.locate_entities(domain, edim, locator)
        entity_indices.append(entities)
        entity_markers.append(np.full_like(entities, marker))
        if entities.size > 0:
            marked[key] = marker
    entity_indices = np.hstack(entity_indices).astype(np.int32)
    entity_markers = np.hstack(entity_markers).astype(np.int32)
    sorted_facets = np.argsort(entity_indices)
    mesh_tags = df.mesh.meshtags(
        domain, edim, entity_indices[sorted_facets], entity_markers[sorted_facets]
    )
    return mesh_tags, marked

def define_problem(mat: IncrSmallStrainModel, dim: int):
    """define problem set up for test_plane_strain"""

    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    def right_boundary(x):
        return np.isclose(x[0], 1.0)

    def y0_boundary(x):
        return np.isclose(x[1], 0.0)

    def y1_boundary(x):
        return np.isclose(x[1], 1.0)

    def z0_boundary(x):
        return np.isclose(x[2], 0.0)

    def z1_boundary(x):
        return np.isclose(x[2], 1.0)

    displacement = 0.01

    if dim == 2:
        mesh = df.mesh.create_unit_square(MPI.COMM_WORLD, 1, 1)
        law = mat(
            parameters={"E0": youngs_modulus, "E1": visco_modulus, "tau": relaxation_time, "nu": poissons_ratio},
            constraint=Constraint.PLANE_STRAIN,
        )
        fixed_vector = np.array([0.0, 0.0])
    elif dim == 3:
        mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
        law = mat(
            parameters={"E0": youngs_modulus, "E1": visco_modulus, "tau": relaxation_time, "nu": poissons_ratio},
            constraint=Constraint.FULL,
        )
        fixed_vector = np.array([0.0, 0.0, 0.0])

    V = df.fem.VectorFunctionSpace(mesh, ("CG", 1))
    u = df.fem.Function(V)

    # boundaries
    fdim = mesh.topology.dim - 1
    left_f = df.mesh.locate_entities_boundary(mesh, fdim, left_boundary)
    right_f = df.mesh.locate_entities_boundary(mesh, fdim, right_boundary)
    y1_f = df.mesh.locate_entities_boundary(mesh, fdim, y1_boundary)
    y0_f = df.mesh.locate_entities_boundary(mesh, fdim, y0_boundary)

    fix_left = df.fem.dirichletbc(fixed_vector, df.fem.locate_dofs_topological(V, fdim, left_f), V)
    fix_y1 = df.fem.dirichletbc(
        df.fem.Constant(mesh, 0.0),
        df.fem.locate_dofs_topological(V.sub(1), fdim, y1_f),
        V.sub(1),
    )
    fix_y0 = df.fem.dirichletbc(
        df.fem.Constant(mesh, 0.0),
        df.fem.locate_dofs_topological(V.sub(1), fdim, y0_f),
        V.sub(1),
    )

    move_ux_right = df.fem.dirichletbc(
        df.fem.Constant(mesh, displacement),
        df.fem.locate_dofs_topological(V.sub(0), fdim, right_f),
        V.sub(0),
    )
    if dim == 2:
        bc_list = [fix_left, fix_y1, fix_y0, move_ux_right]
    elif dim == 3:
        z0_f = df.mesh.locate_entities_boundary(mesh, fdim, z0_boundary)
        z1_f = df.mesh.locate_entities_boundary(mesh, fdim, z1_boundary)
        fix_z0 = df.fem.dirichletbc(
            df.fem.Constant(mesh, 0.0),
            df.fem.locate_dofs_topological(V.sub(2), fdim, z0_f),
            V.sub(2),
        )
        fix_z1 = df.fem.dirichletbc(
            df.fem.Constant(mesh, 0.0),
            df.fem.locate_dofs_topological(V.sub(2), fdim, z1_f),
            V.sub(2),
        )
        bc_list = [fix_left, fix_y0, fix_y1, fix_z0, fix_z1, move_ux_right]

    #problem
    problem = IncrSmallStrainProblem(
        law,
        u,
        bc_list,
        1,
    )
    return u, problem

@pytest.mark.parametrize("mat", [SpringKelvinModel, SpringMaxwellModel])
def test_plane_strain(mat: IncrSmallStrainModel):
    '''compare 3D with fixed z direction with 2D plane strain'''

    u_2D, problem_2D = define_problem(mat, 2)
    solver_2D = NewtonSolver(MPI.COMM_WORLD, problem_2D)

    u_3D, problem_3D = define_problem(mat, 3)
    solver_3D = NewtonSolver(MPI.COMM_WORLD, problem_3D)


    # set time step and solve until total time
    time = [0]
    dt = 5
    problem_2D._time = dt
    problem_3D._time = dt
    total_time = 20 * relaxation_time
    while time[-1] < total_time:
        time.append(time[-1] + dt)
        _ = solver_2D.solve(u_2D)
        problem_2D.update()

        _ = solver_3D.solve(u_3D)
        problem_3D.update()


        # print('2D',problem_2D.stress_1.x.array[0], problem_2D.stress_1.x.array[1], u_2D.x.array.max())
        # print('3D',problem_3D.stress_1.x.array[0], problem_2D.stress_1.x.array[1], u_3D.x.array.max())

        assert abs(problem_2D.stress_1.x.array[0] - problem_3D.stress_1.x.array[0]) < 1e-8
        assert abs(problem_2D.stress_1.x.array[1] - problem_3D.stress_1.x.array[1]) < 1e-8
        assert abs(u_2D.x.array.max() - u_3D.x.array.max()) < 1e-8




if __name__ == "__main__":

    #test_relaxation_uniaxial_stress(SpringKelvinModel)
    #test_relaxation_uniaxial_stress(SpringMaxwellModel)
    #
    # test_relaxation({'dim': 2, 'constraint':'plane_stress'}, SpringMaxwellModel)
    # test_relaxation({'dim': 2, 'constraint':'plane_stress'}, SpringKelvinModel)
    #
    # test_relaxation({'dim': 3}, SpringMaxwellModel)
    # test_relaxation({'dim': 3}, SpringKelvinModel)
    #
    # test_kelvin_vs_maxwell()

    # test_creep({'dim': 3}, SpringKelvinModel)
    # test_creep({'dim': 3}, SpringMaxwellModel)

    # test_creep({'dim': 2, 'constraint':'plane_stress'}, SpringMaxwellModel)
    # test_creep({'dim': 2, 'constraint':'plane_stress'}, SpringKelvinModel)

    test_plane_strain(SpringKelvinModel)