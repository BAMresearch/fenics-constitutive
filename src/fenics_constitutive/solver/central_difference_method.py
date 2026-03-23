from __future__ import annotations

from typing import cast

import basix
import dolfinx as df
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from scipy.linalg import eigvals

from fenics_constitutive.models import StressStrainConstraint
from fenics_constitutive.models.interfaces import IncrSmallStrainModel
from fenics_constitutive.solver._lawonsubmesh import LawOnSubMesh
from fenics_constitutive.solver._solver import SimulationTime
from fenics_constitutive.solver.utils import ufl_mandel_strain

from ._solver import IncrSmallStrainProblem


class CDMSolver:
    r"""
    Class to solve the incremental small strain problem using the central difference method.
    This class will also determine the critical timestep $\Delta t_\mathrm{crit}$. The user may however prescribe a
    timestep in the `IncrSmallStrainProblem` $\Delta t_\mathrm{problem}$ to be used instead. The solver will then use

    $$
    \Delta t =\min\{s\cdot\Delta t_\mathrm{crit}, \Delta t_\mathrm{problem}\}
    $$
    with $s$ being the safety factor.

    Args:
        problem: The incremental small strain problem to be solved.
        density: The density of the material. This is either a single 
            value for a homogenous domain or a list of values for each submesh.
        u0: The initial displacement. This is used to set the initial condition for the solver.
        v0: The initial velocity. This is used to set the initial condition for the solver.
        safety_factor: The safety factor for the time step. The time step is set to the critical 
            time step multiplied by the safety factor. This should be a value between 0 and 1.
    """

    def __init__(
        self,
        problem: IncrSmallStrainProblem,
        density: list[float] | float,
        u0: df.fem.Function,
        v0: df.fem.Function,
        safety_factor: float,
    ) -> None:
        self.problem = problem
        
        density = [density] if isinstance(density, float) else density

        laws = [(law.law, law.cells) for law in problem._law_on_submeshs]
        
        assert len(density) == len(laws)

        self.del_t_crit = critical_timestep(laws, density, u0)

        self.problem.sim_time.set_timestep(self.del_t_crit.min() * safety_factor)
        
        self.problem.incr_disp.current.x.array[:] = u0.x.array[:]
        self.problem.incr_disp.current.x.scatter_forward()
        self.problem.incr_disp.previous.x.array[:] = u0.x.array[:]
        self.problem.incr_disp.previous.x.scatter_forward()

        self.f = cast(df.fem.Function, df.fem.Function(v0.function_space))
        self.a = cast(df.fem.Function, df.fem.Function(v0.function_space))
        self.v = v0
        cells = [law[1] for law in laws]
        self.M_inv = diagonal_inverted_mass(
            v0.function_space, density, cells
        )

    def step(self) -> None:
        r"""Advance the solution by $\Delta t$"""

        df.fem.assemble_vector(self.f.x.array, self.problem.L)
        self.f.x.scatter_reverse(df.la.InsertMode.add)

        self.a.x.array[:] = self.M_inv.x.array * self.f.x.array
        self.a.x.scatter_forward()
        
        self.v.x.array[:] += self.problem.sim_time.dt * self.a.x.array
        self.v.x.scatter_forward()
        
        self.problem.incr_disp.current.x.array[:] += self.problem.sim_time.dt * self.v.x.array
        for bc in self.problem.bcs:
            bc.set(self.problem.incr_disp.current.x.array)
        self.problem.incr_disp.current.x.scatter_forward()
        
        for bc in self.problem.bcs:
            bc.set(self.v.x.array, self.problem.incr_disp.previous.x.array, 1.0/self.problem.sim_time.dt)
        self.v.x.scatter_forward()

        self.problem.form_without_petsc(evaluate_tangent=False)


def critical_timestep(
    laws: list[tuple[IncrSmallStrainModel, np.ndarray]],
    density: list[float],
    u: df.fem.Function,
    method: str = "cdm",
    h: float | None = None,
) -> np.ndarray:
    r"""
    Determines the critical timesteps for all submeshes. This assumes that the constitutive law
    returns a linear elastic tangent for $\sigma=0,\varepsilon=0$. The input is not verified for
    consistency as this function is supposed to be called in the CDMSolver or any other solver.
    
    Args:
        laws: A list of tuples containing the constitutive law and the corresponding cells for each submesh.
        density: A list of densities for each submesh.
        u: The current displacement. This is used to determine the geometric dimension and the function space of the problem.
        method: The time integration method. This is used to determine the factor for the critical time step. Currently only 
            "cdm" is implemented, which corresponds to the central difference method. The factor for the central difference method is 2,
            which means that the time step should be halved.
        h: The mesh size. If this is not provided, the mesh size will be determined based on the mesh and the cells for each submesh.
            This can be used to override the mesh size if the user already knows `h` for example from the mesh creation.
    
    Returns:
        An array containing the critical timesteps of all submeshes.
    """
    method_to_factor = {"cdm": 2}
    factor = method_to_factor[method]
    mesh = u.function_space.mesh

    del_t: list[float] = []
    for (law, cells), density_ in zip(laws, density):
        tangent = np.zeros((law.stress_strain_dim, law.stress_strain_dim))
        stress = np.zeros(law.stress_strain_dim)
        grad_del_u = np.zeros((law.geometric_dim, law.geometric_dim))
        history = (
            {key: np.zeros(dim) for (key, dim) in law.history_dim.items()}
            if law.history_dim is not None
            else None
        )
        law.evaluate(
            0.0,
            1e-12,
            grad_del_u,
            stress,
            tangent.reshape(
                -1,
            ),
            history,
        )
        assert np.linalg.norm(tangent) > 0.0, (
            "The constitutive law must return a non-zero tangent"
        )
        h_ = mesh.h(mesh.topology.dim, cells) if h is None else np.array([h])
        h_min = h_.min()
        omega = _max_frequency_one_element(h_min, u, tangent, density_, law.constraint)
        del_t.append(factor / omega)
    return np.array(del_t)


def _max_frequency_one_element(
    h: float,
    u: df.fem.Function,
    tangent: np.ndarray,
    density: float,
    constraint: StressStrainConstraint,
) -> float:
    """
    Determines the maximum frequency for one element based on the tangent stiffness and the density. 
    This is used to determine the critical time step for the central difference method.

    Args:
        h: The mesh size. This is used to determine the stiffness of the element.
        u: The current displacement. This is used to determine the function space of the problem.
        tangent: The tangent stiffness of the material. This is used to determine the stiffness of the element.
        density: The density of the material. This is used to determine the mass of the element.
        constraint: The stress-strain constraint of the material. This is used to determine the ufl form of the stiffness matrix.
    
    Returns:
        The maximumm frequency of a representative element.
    """
    mesh = u.function_space.mesh
    mesh_cell = mesh.ufl_cell().cellname()

    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    V_degree = u.function_space.ufl_element().degree

    match mesh_cell:
        case "interval":
            h_mesh = df.mesh.create_interval(MPI.COMM_SELF, 1, np.array([0.0, h]))
        case "triangle" | "quadrilateral":
            h_mesh = df.mesh.create_rectangle(
                MPI.COMM_SELF,
                [[0.0, 0.0], [h, h]],
                [1, 1],
                cell_type=df.mesh.CellType[mesh_cell],
            )
        case "tetrahedron" | "hexahedron":
            h_mesh = df.mesh.create_box(
                MPI.COMM_SELF,
                [[0.0, 0.0, 0.0], [h, h, h]],
                [1, 1, 1],
                cell_type=df.mesh.CellType[mesh_cell],
            )
        case _:
            msg = f"Celltype {mesh_cell} not implemented"
            raise Exception(msg)
    V_h = df.fem.functionspace(h_mesh, ("CG", V_degree, (h_mesh.geometry.dim,)))
    h_u, h_v = ufl.TrialFunction(V_h), ufl.TestFunction(V_h)
    tangent_ufl = ufl.as_matrix(tangent.tolist())

    K_form = df.fem.form(
        ufl.inner(
            ufl.dot(tangent_ufl, ufl_mandel_strain(h_u, constraint)),
            ufl_mandel_strain(h_v, constraint),
        )
        * ufl.dx
    )

    h_K = df.fem.assemble_matrix(K_form)

    h_K.scatter_reverse()
    h_K = h_K.to_dense()
    h_M_diag = diagonal_inverted_mass(V_h, [density], [np.arange(num_cells)])
    h_M_diag = np.diagflat(1.0 / h_M_diag.x.array)

    max_eig = float(np.linalg.norm(eigvals(h_K, h_M_diag), np.inf))
    # TODO gather results from all ranks
    return max_eig**0.5


def diagonal_inverted_mass(
    function_space: df.fem.FunctionSpace, density: list[float], cells: list[np.ndarray]
) -> df.fem.Function:
    """
    Determine the inverse of the diagonal mass matrix. For intervals, quadrilaterals and hexahedra, the
    Gauss-Lobatto-Legendre quadrature is used to compute the mass matrix, which results in a diagonal mass matrix.
    For other cell types, this function is not implemented and an exception is raised.

    Args:
        function_space: The function space for which the mass matrix is computed.
        density: The density of the material. This is either a single value for a homogenous domain or a list of values for each submesh.
        cells: The cells corresponding to each submesh. This is used to assign the density to the correct cells in the mass matrix assembly.
    
    Returns:
        The inverted diagonal mass matrix as a function object.
    """
    mesh_cell = function_space.mesh.ufl_cell().cellname()
    basix_cell = basix.CellType[mesh_cell]
    if basix_cell in [
        basix.CellType.interval,
        basix.CellType.quadrilateral,
        basix.CellType.hexahedron,
    ]:
        # do gll integration
        # todo:adapt for higher order elements
        p_degree_to_q_degree = {1: 1, 2: 2}
        geo_dim = function_space.mesh.geometry.dim
        V_degree = function_space.ufl_element().degree
        action_fn = cast(df.fem.Function, df.fem.Function(function_space))
        action_fn.x.array[:] = 1.0
        action_fn.x.scatter_forward()
        q_degree = p_degree_to_q_degree[V_degree]

        metadata = {"quadrature_degree": q_degree, "quadrature_scheme": "gll"}
        dxm = ufl.dx(metadata=metadata)
        if len(density) > 1:
            density_space = df.fem.functionspace(function_space.mesh, ("DG", 0))
            density_fn = cast(df.fem.Function, df.fem.Function(density_space))
            for density_, cells_ in zip(density, cells):
                density_fn.x.array[cells_] = density_
            density_fn.x.scatter_forward()
        else:
            density_fn = df.fem.Constant(function_space.mesh, density[0])

        u_ = ufl.TestFunction(function_space)
        v_ = ufl.TrialFunction(function_space)
        mass_form = cast(
            ufl.Form, ufl.action(density_fn * ufl.inner(u_, v_) * dxm, action_fn)
        )
        M_action = cast(df.fem.Function, df.fem.Function(function_space))
        df.fem.assemble_vector(M_action.x.array, df.fem.form(mass_form))
        M_action.x.scatter_reverse(df.la.InsertMode.add)
    else:
        raise Exception(
            "Only implemented for intervals, quadrilaterals and hexahedral elements"
        )
    M_action.x.array[:] = 1.0 / M_action.x.array[:]
    M_action.x.scatter_forward()
    return M_action
