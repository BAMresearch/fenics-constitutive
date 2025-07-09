from __future__ import annotations

import adios4dolfinx
import dolfinx as df
import numpy as np
import pytest
import ufl
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI

from fenics_constitutive import IncrSmallStrainProblem, norm
from fenics_constitutive.models import VonMises3D
from fenics_constitutive.solver._problemdescription import (
    IncrSmallStrainProblemDescription,
)


def uniaxial_strain_3d_fine_mesh(comm, mesh_path):
    # Fine mesh with more cells
    with df.io.XDMFFile(comm, mesh_path, "r") as xdmf:
        mesh = xdmf.read_mesh()
    V = df.fem.functionspace(mesh, ("CG", 1, (3,)))
    u = df.fem.Function(V, name="Displacement")
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

    def rot_1(x):
        return np.isclose(x[1], 0.0)

    def rot_2(x):
        return np.isclose(x[2], 0.0)

    tdim = mesh.topology.dim
    fdim = tdim - 1

    # ### Dirichlet BCs
    zero_scalar = df.fem.Constant(mesh, np.array([0.0, 0.0, 0.0]))
    scalar_x = df.fem.Constant(mesh, np.array([0.015, 0.0, 0.0]))
    dofs_left = df.fem.locate_dofs_geometrical(V, left)
    dofs_right = df.fem.locate_dofs_geometrical(V, right)
    bc_left = df.fem.dirichletbc(zero_scalar, dofs_left, V)
    bc_right = df.fem.dirichletbc(scalar_x, dofs_right, V)
    dirichlet = [bc_left, bc_right]
    description = IncrSmallStrainProblemDescription(
        laws=law,
        displacement_field=u,
        quadrature_degree=1,
    )
    description.add_boundary_conditions(*dirichlet)
    problem = description.to_problem()

    solver = NewtonSolver(comm, problem)
    nTime = 100
    max_disp = 0.05
    load_steps = np.linspace(0, 1, num=nTime + 1)[1:]
    iterations = np.array([], dtype=np.int32)
    displacement = [0.0]
    load = [0.0]

    for inc, time in enumerate(load_steps):
        current_disp = time * max_disp
        scalar_x.value = np.array([current_disp, 0.0, 0.0])

        niter, converged = solver.solve(u)
        problem.update()

        iterations = np.append(iterations, niter)

        stress_values = []
        stress_values.append(problem.stress_0.x.array.copy())
        stress_values = stress_values[0]
        stress_values = stress_values[::6][0]

        displacement.append(current_disp)
        load.append(stress_values)

    # write results to file
    comm_name = "self" if comm == MPI.COMM_SELF else "world"
    adios4dolfinx.write_function_on_input_mesh(
        f"mpi_test_{comm_name}.bp",
        u,
        mode=adios4dolfinx.adios2_helpers.adios2.Mode.Write,
    )
    return f"mpi_test_{comm_name}.bp"


@pytest.mark.mpi
def test_mpi_solver():
    """
    Test the MPI solver by comparing results from two different communicators.
    """
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 4, 6, 7)
    with df.io.XDMFFile(MPI.COMM_WORLD, "fine_mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
    name_world = uniaxial_strain_3d_fine_mesh(MPI.COMM_WORLD, "fine_mesh.xdmf")
    name_self = uniaxial_strain_3d_fine_mesh(MPI.COMM_SELF, "fine_mesh.xdmf")

    with df.io.XDMFFile(MPI.COMM_WORLD, "fine_mesh.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh()
    V = df.fem.functionspace(mesh, ("CG", 1, (3,)))
    u_self = df.fem.Function(V, name="Displacement")
    u_world = df.fem.Function(V, name="Displacement")
    adios4dolfinx.read_function(name_self, u_self)
    adios4dolfinx.read_function(name_world, u_world)
    # Compare the two functions
    diff = df.fem.Function(V, name="Difference")
    diff.x.array[:] = u_self.x.array - u_world.x.array
    diff.x.scatter_forward()

    l_2_norm_diff = norm(diff, ufl.dx, norm_type="l2")
    l_2_norm_rel = norm(u_self, ufl.dx, norm_type="l2")
    print(f"L2 norm of the difference: {l_2_norm_diff}")
    print(f"Relative L2 norm of the difference: {l_2_norm_diff / l_2_norm_rel}")
    assert l_2_norm_diff < 1e-14 * l_2_norm_rel, (
        f"The L2 norm of the difference is {l_2_norm_diff / l_2_norm_rel}. It should be smaller than 1e-14."
    )
