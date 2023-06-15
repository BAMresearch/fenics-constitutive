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
    
    def create_quadrature_space_like(self, function_space : dfx.fem.FunctionSpace):
        element = function_space.ufl_element()
        if len(element.value_shape()) == 0:
            return self.create_quadrature_space(function_space.mesh)
        elif len(element.value_shape()) == 1:
            return self.create_quadrature_vector_space(function_space.mesh, element.value_shape()[0])
        elif len(element.value_shape()) == 2:
            return self.create_quadrature_tensor_space(function_space.mesh, element.value_shape())

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
    if cell_type in [basix.CellType.interval,basix.CellType.quadrilateral, basix.CellType.hexahedron]:
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

class TimestepEstimator:
    def __init__(self, mesh, G, K, rho, safety_factor=1., order = 1):
        self.del_t = 0.
        self.eig_max = 0.
        self.safety_factor=safety_factor
        self.mesh = mesh
        self.cell_type = self.mesh.topology.cell_type
        if self.cell_type == dfx.mesh.CellType.triangle:
            raise TypeError('Cell type "' + str(self.cell_type) + '" is not yet supported')
        elif self.cell_type == dfx.mesh.CellType.quadrilateral:
            self.h_mesh = dfx.mesh.create_unit_square(
                MPI.COMM_SELF,
                1,
                1,
                cell_type=self.cell_type,
            )
        else:
            raise TypeError('Cell type "' + str(self.cell_type) + '" is not yet supported')

        self.G = dfx.fem.Constant(self.h_mesh, G)
        self.K = dfx.fem.Constant(self.h_mesh, K)
        fdim = self.mesh.topology.dim
        self.mesh.topology.create_connectivity(fdim, 0)

        num_cells_owned_by_proc = self.mesh.topology.index_map(fdim).size_local
        self.cells = dfx.cpp.mesh.entities_to_geometry(self.mesh, fdim, np.arange(num_cells_owned_by_proc, dtype=np.int32), False)
        
        def eps(v):
            return ufl.sym(ufl.grad(v))

        def sigma(v):
            e = eps(v)
            return (self.K - (2.0 / 3.0) * self.G) * ufl.tr(e) * ufl.Identity(2) + 2.0 * self.G * e

        h_P1 = dfx.fem.VectorFunctionSpace(self.h_mesh, ("CG", order))
        h_P1s = dfx.fem.FunctionSpace(self.h_mesh, ("CG", order))
        h_u, h_v = ufl.TrialFunction(h_P1), ufl.TestFunction(h_P1)
        self.K_form = dfx.fem.form(ufl.inner(eps(h_u), sigma(h_v)) * ufl.dx)
        self.M_form = dfx.fem.form(rho * ufl.inner(h_u, h_v) * ufl.dx)
        one = dfx.fem.Function(h_P1s)
        one.x.array[:] = np.ones_like(one.x.array,dtype=np.float64)
        self.V_form = dfx.fem.form(one * ufl.dx)
        self.h_cell = dfx.cpp.mesh.entities_to_geometry(self.h_mesh, fdim, np.arange(1, dtype=np.int32), False)
        #points = self.mesh.geometry.x
        #for e, entity in enumerate(geometry_entitites):
        #    print(e, points[entity])
    def __call__(self, G=None, K=None):
        self.G.value = G if G is not None else self.G.value
        self.K.value = K if K is not None else self.K.value
        h_K, h_M = (
            dfx.fem.petsc.assemble_matrix(self.K_form),
            dfx.fem.petsc.assemble_matrix(self.M_form),
        )
        for cell in self.cells:
            h_K.zeroEntries()
            h_M.zeroEntries()
            self.h_mesh.geometry.x[self.h_cell] = self.mesh.geometry.x[cell]
            dfx.fem.petsc.assemble_matrix(h_K, self.K_form)
            dfx.fem.petsc.assemble_matrix(h_M, self.M_form)
            h_K.assemble()
            h_M.assemble()
            h_M_array = np.array(h_M[:, :])
            h_K_array = np.array(h_K[:, :])
            eig_temp = np.linalg.norm(eigvals(h_K_array, h_M_array), np.inf)
            self.eig_max = max(eig_temp, self.eig_max)
        self.del_t =self.safety_factor *  2. / self.eig_max ** 0.5
        return self.del_t

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

# uncomment once cell_avg is supported in dolfinx
# def b_bar_strain(u):
#     eps = ufl.sym(ufl.grad(u))
#     vol = (1/3) * ufl.cell_avg(ufl.tr(eps))
#     return eps +(vol - (1/3) * ufl.tr(eps)) * ufl.Identity(2)
#def critical_timestep_lanczos():
#    from slepc4py import SLEPc
#    from petsc4py import PETSc
#    PETSc.KSP()