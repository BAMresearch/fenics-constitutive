import numpy as np
import ufl
from mpi4py import MPI
from scipy.linalg import eigvals

import basix
import dolfinx as dfx


def set_mesh_coordinates(mesh, x, mode="set"):
    dim = mesh.geometry.dim
    array = x if type(x) == np.ndarray else x.x.array
    if mode == "set":
        mesh.geometry.x[:, :dim] = array.reshape(-1, dim)
    elif mode == "add":
        mesh.geometry.x[:, :dim] += array.reshape(-1, dim)


def get_local(u, ghost_nodes=False):
    return u.x.array if ghost_nodes else u.vector.array


def set_local(u, u_array):
    u.vector.array[:] = u_array
    u.x.scatter_forward()
    #u.vector.ghostUpdate()


def add_local(u, u_array):
    u.vector.array[:] += u_array
    u.x.scatter_forward()
    #u.vector.ghostUpdate()


class QuadratureRule:
    def __init__(
        self,
        quadrature_type=basix.QuadratureType.Default,
        cell_type=basix.CellType.triangle,
        degree=1,
    ):
        self.quadrature_type = quadrature_type
        self.cell_type = cell_type
        self.degree = degree
        self.points, self.weights = basix.make_quadrature(
            self.quadrature_type, self.cell_type, self.degree
        )
        self.dx = ufl.dx(
            metadata={
                "quadrature_rule": self.quadrature_type.name,
                "quadrature_degree": self.degree,
            }
        )

    def create_quadrature_space(self, mesh):
        Qe = ufl.FiniteElement(
            "Quadrature",
            basix_cell_type_to_ufl(self.cell_type),
            self.degree,
            quad_scheme=self.quadrature_type.name,
        )

        return dfx.fem.FunctionSpace(mesh, Qe)

    def create_quadrature_vector_space(self, mesh, dim):
        Qe = ufl.VectorElement(
            "Quadrature",
            basix_cell_type_to_ufl(self.cell_type),
            self.degree,
            quad_scheme=self.quadrature_type.name,
            dim=dim,
        )

        return dfx.fem.FunctionSpace(mesh, Qe)

    def create_quadrature_tensor_space(self, mesh, shape):
        Qe = ufl.TensorElement(
            "Quadrature",
            basix_cell_type_to_ufl(self.cell_type),
            self.degree,
            quad_scheme=self.quadrature_type.name,
            shape=shape,
        )

        return dfx.fem.FunctionSpace(mesh, Qe)
        
    def number_of_points(self, mesh):
        map_c = mesh.topology.index_map(mesh.topology.dim)
        self.num_cells = map_c.size_local
        return self.num_cells * self.weights.size


def basix_cell_type_to_ufl(cell_type: basix.CellType) -> ufl.Cell:
    conversion = {
        basix.CellType.interval: ufl.interval,
        basix.CellType.triangle: ufl.triangle,
        basix.CellType.tetrahedron: ufl.tetrahedron,
        basix.CellType.quadrilateral: ufl.quadrilateral,
        basix.CellType.hexahedron: ufl.hexahedron,
    }
    return conversion[cell_type]


class QuadratureEvaluator:
    def __init__(self, ufl_expression, mesh, quadrature_rule):
        map_c = mesh.topology.index_map(mesh.topology.dim)
        self.num_cells = map_c.size_local  # + map_c.num_ghosts
        try:
            assert map_c.num_ghosts == 0
        except AssertionError as e:
            print(
                f"Warning: In QuadratureEvaluator: There are {map_c.num_ghosts} Quadrature ghost points."
            )

        self.cells = np.arange(0, self.num_cells, dtype=np.int32)

        self.expr = dfx.fem.Expression(ufl_expression, quadrature_rule.points)

    def __call__(self, q=None):
        if q is None:
            return self.expr.eval(self.cells)
        elif type(q) == np.ndarray:
            self.expr.eval(self.cells, values=q.reshape(self.num_cells, -1))
        else:
            self.expr.eval(
                self.cells, values=q.vector.array.reshape(self.num_cells, -1)
            )
            q.x.scatter_forward()


def project(v, V, dx, u=None):
    dv = ufl.TrialFunction(V)
    v_ = ufl.TestFunction(V)
    a_proj = ufl.inner(dv, v_) * dx
    b_proj = ufl.inner(v, v_) * dx
    if u is None:
        solver = dfx.fem.petsc.LinearProblem(a_proj, b_proj)
        uh = solver.solve()
        return uh
    else:
        solver = dfx.fem.petsc.LinearProblem(a_proj, b_proj, u=u)
        solver.solve()


def diagonal_mass(
    function_space, rho, cell_type=basix.CellType.quadrilateral, invert=True
):
    if cell_type == basix.CellType.quadrilateral:
        # do gll integration
        # todo:adapt for higher order elements
        V_degree = function_space.ufl_element().degree()
        degree = 1 if V_degree==1 else 2
        rule = QuadratureRule(
            quadrature_type=basix.QuadratureType.gll, cell_type=cell_type, degree=degree
        )
        u_ = ufl.TestFunction(function_space)
        v_ = ufl.TrialFunction(function_space)
        mass_form = ufl.inner(u_, v_) * rho * rule.dx

        M = dfx.fem.petsc.assemble_matrix(dfx.fem.form(mass_form))
        M.assemble()
        M_action = M.getDiagonal()
    else:
        rule = QuadratureRule(
            quadrature_type=basix.QuadratureType.Default, cell_type=cell_type, degree=1
        )
        u_ = ufl.TestFunction(function_space)
        v_ = ufl.TrialFunction(function_space)
        mass_form = ufl.inner(u_, v_) * rho * rule.dx
        v_temp = dfx.fem.Function(function_space)
        ones = v_temp.vector.copy()
        # mass_action = dfx.fem.form(ufl.action(mass_form, ))

        M = dfx.fem.petsc.assemble_matrix(dfx.fem.form(mass_form))
        M.assemble()
        with ones.localForm() as ones_local:
            ones_local.set(1.0)
        M_action = M * ones
    if invert:
        M_action.array[:] = 1.0 / M_action.array
        M_action.ghostUpdate()
    return M_action


def critical_timestep(mesh, l_x, l_y, G, K, rho, order=1):
    # todo: implement other cell_types
    cell_type = mesh.topology.cell_type
    if cell_type == dfx.mesh.CellType.triangle:
        h_mesh = dfx.mesh.create_rectangle(
            MPI.COMM_SELF,
            np.array([[0.0, 0.0], [l_x, l_y]]),
            [1, 1],
            cell_type=cell_type,
            diagonal=DiagonalType.crossed,
        )
    elif cell_type == dfx.mesh.CellType.quadrilateral:
        h_mesh = dfx.mesh.create_rectangle(
            MPI.COMM_SELF,
            np.array([[0.0, 0.0], [l_x, l_y]]),
            [1, 1],
            cell_type=cell_type,
        )
    else:
        raise TypeError('Cell type "' + str(cell_type) + '" is not yet supported')

    def eps(v):
        return ufl.sym(ufl.grad(v))

    def sigma(v):
        e = eps(v)
        return (K - (2.0 / 3.0) * G) * ufl.tr(e) * ufl.Identity(2) + 2.0 * G * e

    h_P1 = dfx.fem.VectorFunctionSpace(h_mesh, ("CG", order))
    h_u, h_v = ufl.TrialFunction(h_P1), ufl.TestFunction(h_P1)
    K_form = dfx.fem.form(ufl.inner(eps(h_u), sigma(h_v)) * ufl.dx)
    M_form = dfx.fem.form(rho * ufl.inner(h_u, h_v) * ufl.dx)

    h_K, h_M = (
        dfx.fem.petsc.assemble_matrix(K_form),
        dfx.fem.petsc.assemble_matrix(M_form),
    )
    h_K.assemble()
    h_M.assemble()
    h_M = np.array(h_M[:, :])
    h_K = np.array(h_K[:, :])
    max_eig = np.linalg.norm(eigvals(h_K, h_M), np.inf)

    h = 2.0 / max_eig ** 0.5
    return h


def critical_timestep_2(l_x, l_y, G, K, rho, cell_type=dfx.mesh.CellType.quadrilateral, order=1):
    # todo: implement other cell_types
    # cell_type=mesh.topology.cell_type
    if cell_type == dfx.mesh.CellType.triangle:
        h_mesh = dfx.mesh.create_rectangle(
            MPI.COMM_SELF,
            np.array([[0.0, 0.0], [l_x, l_y]]),
            [1, 1],
            cell_type=cell_type,
            diagonal=DiagonalType.crossed,
        )
    elif cell_type == dfx.mesh.CellType.quadrilateral:
        h_mesh = dfx.mesh.create_rectangle(
            MPI.COMM_SELF,
            np.array([[0.0, 0.0], [l_x, l_y]]),
            [1, 1],
            cell_type=cell_type,
        )
    else:
        raise TypeError('Cell type "' + str(cell_type) + '" is not yet supported')

    def eps(v):
        return ufl.sym(ufl.grad(v))

    def sigma(v):
        e = eps(v)
        return (K - (2.0 / 3.0) * G) * ufl.tr(e) * ufl.Identity(2) + 2.0 * G * e

    h_P1 = dfx.fem.VectorFunctionSpace(h_mesh, ("CG", order))
    h_u, h_v = ufl.TrialFunction(h_P1), ufl.TestFunction(h_P1)
    K_form = dfx.fem.form(ufl.inner(eps(h_u), sigma(h_v)) * ufl.dx)
    M_form = dfx.fem.form(rho * ufl.inner(h_u, h_v) * ufl.dx)

    h_K, h_M = (
        dfx.fem.petsc.assemble_matrix(K_form),
        dfx.fem.petsc.assemble_matrix(M_form),
    )
    h_K.assemble()
    h_M.assemble()
    h_M = np.array(h_M[:, :])
    h_K = np.array(h_K[:, :])
    max_eig = np.linalg.norm(eigvals(h_K, h_M), np.inf)

    h = 2.0 / max_eig ** 0.5
    return h


def critical_timestep_nonlocal(
    l_x, l_y, l, zeta, cell_type=dfx.mesh.CellType.quadrilateral, rule=None, order=1
):
    # todo: implement other cell_types
    if cell_type == dfx.mesh.CellType.triangle:
        h_mesh = dfx.mesh.create_rectangle(
            MPI.COMM_SELF,
            np.array([[0.0, 0.0], [l_x, l_y]]),
            [1, 1],
            cell_type=cell_type,
            diagonal=DiagonalType.crossed,
        )
    elif cell_type == dfx.mesh.CellType.quadrilateral:
        h_mesh = dfx.mesh.create_rectangle(
            MPI.COMM_SELF,
            np.array([[0.0, 0.0], [l_x, l_y]]),
            [1, 1],
            cell_type=cell_type,
        )
    else:
        raise TypeError('Cell type "' + str(cell_type) + '" is not yet supported')
    dx = rule.dx if rule is not None else ufl.dx
    h_P1 = dfx.fem.FunctionSpace(h_mesh, ("CG", order))
    h_u, h_v = ufl.TrialFunction(h_P1), ufl.TestFunction(h_P1)
    K_form = dfx.fem.form(
        (l ** 2 * ufl.inner(ufl.grad(h_u), ufl.grad(h_v)) + h_u * h_v) * ufl.dx
    )
    M_form = dfx.fem.form(zeta * ufl.inner(h_u, h_v) * ufl.dx)

    h_K, h_M = (
        dfx.fem.petsc.assemble_matrix(K_form),
        dfx.fem.petsc.assemble_matrix(M_form),
    )
    h_K.assemble()
    h_M.assemble()
    h_M = np.array(h_M[:, :])
    h_K = np.array(h_K[:, :])

    max_eig = np.linalg.norm(eigvals(h_K, h_M), np.inf)

    h = 2.0 / max_eig ** 0.5
    return h
