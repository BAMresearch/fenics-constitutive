from __future__ import annotations

import dolfinx as df
import numpy as np
from dolfinx.nls.petsc import NewtonSolver
from mises_plasticity_isotropic_hardening import VonMises3D
from mpi4py import MPI

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from fenics_constitutive import Constraint, IncrSmallStrainProblem


def test_3d(): # uniaxial strain 3d
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    V = df.fem.VectorFunctionSpace(mesh, ("CG", 1))
    u = df.fem.Function(V)
    matparam = {
        "p_ka": 175000,
        "p_mu": 80769,
        "p_y0": 1200,
        "p_y00":2500,
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
    #
    problem = IncrSmallStrainProblem(law, u, dirichlet)

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

    assert load[-1] <= matparam["p_y00"]
    indices = load <= matparam["p_y0"]
    v = (3 * 175000 - 2 * 80769) / (2 * (3 * 175000 + 80769))
    trace = displacement[indices][1] - 2 * v * displacement[indices][1]
    dev = displacement[indices][1] - trace / 3
    #print((175000 * trace + 2 * 80769 * dev) / displacement[indices][1])
    assert abs((load[indices][1] / displacement[indices][1]) - ((175000 * trace + 2 * 80769 * dev) / displacement[indices][1])) < 1e-8


    ax = plt.subplots()[1]
    # ax.plot(sol.eps, sol.sigma, "r-", label="analytical")
    ax.plot(displacement, load, label="numerical")
    ax.set_xlabel(r"$\varepsilon_{xx}$")
    ax.set_ylabel(r"$\sigma_{xx}$")
    ax.legend()
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    plt.show()

if __name__ == "__main__":
    test_3d()