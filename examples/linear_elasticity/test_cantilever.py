from __future__ import annotations

import dolfinx as df
import numpy as np
from dolfinx.nls.petsc import NewtonSolver
# from mises_plasticity_isotropic_hardening import VonMises3D
from mpi4py import MPI
from fenics_constitutive import Constraint, IncrSmallStrainProblem
import matplotlib.pyplot as plt
from linear_elasticity_model import LinearElasticityModel
from dolfinx import log
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
# import pyvista
import numpy as np
import ufl

from mpi4py import MPI
from dolfinx import fem, mesh, plot, io

youngs_modulus = 1.0e4
poissons_ratio = 0.3

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


def test_cantilever_beam():
    L = 20.0
    domain = df.mesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [L, 1, 1]], [20, 5, 5], df.mesh.CellType.hexahedron)
    V = df.fem.VectorFunctionSpace(domain, ("CG", 2))
    # V = fem.VectorFunctionSpace(domain, ("Lagrange", 2,))
    u = df.fem.Function(V)

    def left(x):
        return np.isclose(x[0], 0)

    def right(x):
        return np.isclose(x[0], L)

    fdim = domain.topology.dim - 1
    left_facets = df.mesh.locate_entities_boundary(domain, fdim, left)
    right_facets = df.mesh.locate_entities_boundary(domain, fdim, right)

    # Concatenate and sort the arrays based on facet indices. Left facets marked with 1, right facets with two
    marked_facets = np.hstack([left_facets, right_facets])
    marked_values = np.hstack([np.full_like(left_facets, 1), np.full_like(right_facets, 2)])
    sorted_facets = np.argsort(marked_facets)
    facet_tag = df.mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

    u_bc = np.array((0,) * domain.geometry.dim, dtype=np.float64)

    left_dofs = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.find(1))
    bcs = [fem.dirichletbc(u_bc, left_dofs, V)]

    law = LinearElasticityModel(
        parameters={"E": youngs_modulus, "nu": poissons_ratio},
        constraint=Constraint.FULL,
    )

    neumann_tag = 15
    neumann_boundary = {"right": (neumann_tag, right)}
    facet_tags, _ = create_meshtags(domain, fdim, neumann_boundary)
    dA = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    max_load = -1.5
    neumann_data = df.fem.Constant(domain, (0.0, 0.0, max_load))

    problem = IncrSmallStrainProblem(law, u, bcs, q_degree=4, mesh_update=False, co_rotation=False)

    if problem.mesh_update:
        print("Mesh update is enabled.")
    else:
        print("Mesh update is disabled.")

    if problem.co_rotation:
        print("Co-Rotation is enabled.")
    else:
        print("Co-Rotation is disabled.")

    test_function = ufl.TestFunction(u.function_space)
    fext = ufl.inner(neumann_data, test_function) * dA(neumann_tag)
    problem.R_form -= fext

    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.atol = 1e-8
    solver.rtol = 1e-8
    solver.convergence_criterion = "incremental"

    log.set_log_level(log.LogLevel.INFO)
    tval0 = -1.5
    with io.XDMFFile(domain.comm, "Linear_Elasticity.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        for n in np.linspace(1, 10, 50,endpoint=False):
            neumann_data.value[2] = n * tval0
            num_its, converged = solver.solve(u)
            assert (converged)
            problem.update()
            u.x.scatter_forward()
            print(f"Time step {n}, Number of iterations {num_its}, Load {neumann_data.value}")

            u.name = "Deformation_linear_elasticity"
            xdmf.write_function(u, float(n))  # The second argument can be used to mark the timestep in the XDMF file
