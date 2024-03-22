from mises_plasticity_isotropic_hardening import *
from fenics_constitutive import IncrSmallStrainProblem
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


# The function to run the force-controlled simple tension test.
# We apply a constant force on the right surface, pulling in the
# :math:`\boldsymbol{e}_1`-direction.
def simple_tension_test(mesh, material, pltshow=False):
    function_space = df.fem.VectorFunctionSpace(mesh, ("Lagrange", 1))
    u = df.fem.Function(function_space)

    # ### Definition of BCs for Simple Tension Test
    def origin(x):
        p = [0.0, 0.0, 0.0]
        return np.logical_and(
            np.logical_and(np.isclose(x[0], p[0]), np.isclose(x[1], p[1])),
            np.isclose(x[2], p[2]),
        )

    def x_001(x):
        p = [0.0, 0.0, 1.0]
        return np.logical_and(
            np.logical_and(np.isclose(x[0], p[0]), np.isclose(x[1], p[1])),
            np.isclose(x[2], p[2]),
        )

    def left(x):
        return np.isclose(x[0], 0.0)
    def right(x):
        return np.isclose(x[0], 1.0)

    tdim = mesh.topology.dim
    fdim = tdim - 1

    origin_vertex = df.mesh.locate_entities_boundary(mesh, 0, origin)
    x3_vertex = df.mesh.locate_entities_boundary(mesh, 0, x_001)
    left_facets = df.mesh.locate_entities_boundary(mesh, fdim, left)
    right_facets = df.mesh.locate_entities_boundary(mesh, fdim, right)

    # ### Dirichlet BCs
    zero_scalar = df.fem.Constant(mesh, 0.0)
    zero_scalar_x = df.fem.Constant(mesh, 0.05)
    fix_ux = df.fem.dirichletbc(
        zero_scalar,
        df.fem.locate_dofs_topological(function_space.sub(0), fdim, left_facets),
        function_space.sub(0),
    )
    move_ux = df.fem.dirichletbc(
        zero_scalar_x,
        df.fem.locate_dofs_topological(function_space.sub(0), fdim, right_facets),
        function_space.sub(0),
    )
    fix_uy = df.fem.dirichletbc(
        zero_scalar,
        df.fem.locate_dofs_topological(function_space.sub(1), 0, origin_vertex),
        function_space.sub(1),
    )
    fix_uz = df.fem.dirichletbc(
        zero_scalar,
        df.fem.locate_dofs_topological(function_space.sub(2), 0, origin_vertex),
        function_space.sub(2),
    )
    # rotation around x1-axis
    fix_rot_x1 = df.fem.dirichletbc(
        zero_scalar,
        df.fem.locate_dofs_topological(function_space.sub(1), 0, x3_vertex),
        function_space.sub(1),
    )

    # ### Neumann BCs


    neumann_tag = 15
    neumann_boundary = {"right": (neumann_tag, right)}
    facet_tags, _ = create_meshtags(mesh, fdim, neumann_boundary)
    dA = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    max_load = 0#2718.0
    max_disp = 0.015  # 2718.0
    neumann_data = df.fem.Constant(mesh, (max_load, 0.0, 0.0))

    laws = material
    dirichlet = [fix_ux, move_ux, fix_uy, fix_uz, fix_rot_x1]

    problem = IncrSmallStrainProblem(laws, u, dirichlet)
    # neumann
    test_function = ufl.TestFunction(u.function_space)
    fext = ufl.inner(neumann_data, test_function) * dA(neumann_tag)
    problem.R_form -= fext

    # dirichlet

    # problem.compile(dirichlet)  # optionally add form compiler options


    solver = df.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    import dolfinx.log as log
    log.set_log_level(log.LogLevel.INFO)
    # from dolfinx import log
    # log.set_log_level(log.LogLevel.INFO)
    # ### center point, right surface
    xc_right = np.array([[1.0, 0.5, 0.5]])

    nTime = 100
    #load_steps = np.linspace(0, 1, num=nTime + 1)[1:]
    load_steps = np.linspace(np.pi, -np.pi, num=nTime + 1)
    iterations = np.array([], dtype=np.int32)
    displacement = [0.0]
    load = [0.0]

    for inc, time in enumerate(load_steps):
        print("Load Increment:", inc)

        # external force
        #current_load = time * max_load
        #current_disp = time * max_disp
        current_disp = np.sin(time) * max_disp
        #neumann_data.value = (current_load, 0.0, 0.0)
        neumann_data.value = (0.0, 0.0, 0.0)
        zero_scalar_x.value = (current_disp)

        niter, converged = solver.solve(u)
        assert converged
        problem.update()
        print(f"Converged: {converged} in {niter} iterations.")
        iterations = np.append(iterations, niter)

        # # load displacement data
        # u_right = u.eval(xc_right, cells=problem.cells)
        # displacement.append(u_right.item(0))
        # load.append(current_disp)

        # # load stress data
        # ll = problem.form(dirichlet)
        # load_stress = problem._stress[:].x.array
        # print(load_stress)
        # #u_right = u.eval(xc_right, cells=problem.cells)
        # displacement.append(current_disp)
        # load.append(load_stress)

        stress_values = []
        for k in range(len(problem.laws)):
            stress_values.append(problem._stress[k].x.array.copy())

        stress_values = stress_values[0]
        stress_values = stress_values[::6][0]
        print(stress_values)
        displacement.append(current_disp)
        load.append(stress_values)

    displacement = np.array(displacement)
    load = np.array(load)
    return load, displacement
################################################################################################
# def simple_tension_test_cyclic(mesh, material, pltshow=False):
#     function_space = df.fem.VectorFunctionSpace(mesh, ("Lagrange", 1))
#     u = df.fem.Function(function_space)
#
#     # ### Definition of BCs for Simple Tension Test
#     def origin(x):
#         p = [0.0, 0.0, 0.0]
#         return np.logical_and(
#             np.logical_and(np.isclose(x[0], p[0]), np.isclose(x[1], p[1])),
#             np.isclose(x[2], p[2]),
#         )
#
#     def x_001(x):
#         p = [0.0, 0.0, 1.0]
#         return np.logical_and(
#             np.logical_and(np.isclose(x[0], p[0]), np.isclose(x[1], p[1])),
#             np.isclose(x[2], p[2]),
#         )
#
#     def left(x):
#         return np.isclose(x[0], 0.0)
#
#     tdim = mesh.topology.dim
#     fdim = tdim - 1
#
#     origin_vertex = df.mesh.locate_entities_boundary(mesh, 0, origin)
#     x3_vertex = df.mesh.locate_entities_boundary(mesh, 0, x_001)
#     left_facets = df.mesh.locate_entities_boundary(mesh, fdim, left)
#
#     # ### Dirichlet BCs
#     zero_scalar = df.fem.Constant(mesh, 0.0)
#     fix_ux = df.fem.dirichletbc(
#         zero_scalar,
#         df.fem.locate_dofs_topological(function_space.sub(0), fdim, left_facets),
#         function_space.sub(0),
#     )
#     fix_uy = df.fem.dirichletbc(
#         zero_scalar,
#         df.fem.locate_dofs_topological(function_space.sub(1), 0, origin_vertex),
#         function_space.sub(1),
#     )
#     fix_uz = df.fem.dirichletbc(
#         zero_scalar,
#         df.fem.locate_dofs_topological(function_space.sub(2), 0, origin_vertex),
#         function_space.sub(2),
#     )
#     # rotation around x1-axis
#     fix_rot_x1 = df.fem.dirichletbc(
#         zero_scalar,
#         df.fem.locate_dofs_topological(function_space.sub(1), 0, x3_vertex),
#         function_space.sub(1),
#     )
#
#     # ### Neumann BCs
#     def right(x):
#         return np.isclose(x[0], 1.0)
#
#     neumann_tag = 15
#     neumann_boundary = {"right": (neumann_tag, right)}
#     facet_tags, _ = create_meshtags(mesh, fdim, neumann_boundary)
#     dA = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
#     max_load = 2000#2499.9999#2718.0
#     neumann_data = df.fem.Constant(mesh, (max_load, 0.0, 0.0))
#
#     laws = [(material, None)]
#     problem = RambergOsgoodProblem(laws, u)
#     # neumann
#     test_function = ufl.TestFunction(u.function_space)
#     fext = ufl.inner(neumann_data, test_function) * dA(neumann_tag)
#     problem.R_form -= fext
#
#     # dirichlet
#     dirichlet = [fix_ux, fix_uy, fix_uz, fix_rot_x1]
#     problem.compile(dirichlet)  # optionally add form compiler options
#
#     solver = df.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
#     solver.convergence_criterion = "incremental"
#     # from dolfinx import log
#     # log.set_log_level(log.LogLevel.INFO)
#     # ### center point, right surface
#     xc_right = np.array([[1.0, 0.5, 0.5]])
#
#     nTime = 100
#     load_steps = np.linspace(np.pi, -np.pi, num=nTime + 1)
#     #load_steps = np.linspace(0, 0.91, num=nTime + 1)[1:]
#     iterations = np.array([], dtype=np.int32)
#     displacement = [0.0]
#     load = [0.0]
#
#     for inc, time in enumerate(load_steps):
#         print("Load Increment:", inc)
#
#         # external force
#         current_load = np.sin(time) * max_load
#         if current_load < 0:
#             current_load = current_load*1.2
#         else:
#             pass
#         neumann_data.value = (current_load, 0.0, 0.0)
#         niter, converged = solver.solve(u)
#         assert converged
#         problem.update_history()
#         print(f"Converged: {converged} in {niter} iterations.")
#         iterations = np.append(iterations, niter)
#
#         # load displacement data
#         u_right = u.eval(xc_right, cells=problem.cells)
#         displacement.append(u_right.item(0))
#         load.append(current_load)
#
#
#
#     displacement = np.array(displacement)
#     load = np.array(load)
#     return load, displacement

def main(args):
    n = args.num_cells
    mesh = df.mesh.create_unit_cube(
        MPI.COMM_WORLD, n, n, n, df.mesh.CellType.hexahedron
    )
    matparam = {
        "p_ka": 175000,
        "p_mu": 80769,
        "p_y0": 1200,
        "p_y00":2500,
        "p_w": 200,
    }
    material = VonMises3D(matparam)
    sigma_h, eps_h = simple_tension_test(mesh, material)
    #print(sigma_h)

    # ### Comparison with analytical solution
    # sol = RambergOsgoodSimpleTension(matparam)
    # sol.solve(sigma_h[-1], num_points=51)
    # w = sol.energy()
    # I = np.trapz(sigma_h, eps_h)
    # assert np.isclose((w - I) / w, 0.0, atol=1e-2)

    if args.show:
        ax = plt.subplots()[1]
        #ax.plot(sol.eps, sol.sigma, "r-", label="analytical")
        ax.plot(eps_h, sigma_h, label="numerical")
        ax.set_xlabel(r"$\varepsilon_{xx}$")
        ax.set_ylabel(r"$\sigma_{xx}$")
        ax.legend()
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        plt.show()


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "num_cells",
        type=int,
        help="Number of cells in each spatial direction of the unit cube.",
    )
    parser.add_argument("--show", action="store_true", help="Show plot.")
    args = parser.parse_args(sys.argv[1:])
    main(args)