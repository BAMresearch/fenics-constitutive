import basix
import ufl
from fenics_constitutive.solver.utils import ufl_mandel_strain
from fenics_constitutive.solver.typesafe import fn_for
from fenics_constitutive.solver._incrementalunknowns import IncrementalDisplacement
from __future__ import annotations

import dolfinx as df
import numpy as np
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc
from mpi4py import MPI
from scipy.linalg import eigvals
from fenics_constitutive.models.interfaces import IncrSmallStrainModel
from fenics_constitutive.solver._incrementalunknowns import IncrementalStress
from fenics_constitutive.solver._lawonsubmesh import LawOnSubMesh, create_law_on_submesh
from fenics_constitutive.solver._solver import SimulationTime
from fenics_constitutive.solver._spaces import ElementSpaces

from ._solver import IncrSmallStrainProblem


class CDMSolver:
    """
    Corotational extension of IncrSmallStrainProblem that adds objective
    stress-rate rotation and mesh-update steps.

    This subclass reuses all core functionality of the base solver and
    overrides only the initialization and `form` method.
    """

    def __init__(
        self,
        laws: list[tuple[IncrSmallStrainModel, np.ndarray]] | IncrSmallStrainModel,
        density: list[float] | float,
        u: df.fem.Function,
        v: df.fem.Function,
        bcs: list[df.fem.DirichletBC],
        q_degree: int,
        del_t: float | None = None,
        external_forces: ufl.Form | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
    ) -> None:
        mesh = u.function_space.mesh
        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        if isinstance(laws, IncrSmallStrainModel):
            assert isinstance(density, float)
            density = [density]
            laws = [(laws, np.arange(0, num_cells, dtype=np.int32))]

        assert len(density) == len(laws)

        constraint = laws[0][0].constraint
        assert all(law[0].constraint == constraint for law in laws), (
            "All laws must have the same constraint"
        )

        element_spaces = ElementSpaces.create(mesh, constraint, q_degree)
        self.stress = IncrementalStress(element_spaces.stress_vector_space)
        self.tangent = fn_for(element_spaces.stress_tensor_space(mesh))

        self._law_on_submeshs: list[LawOnSubMesh] = []

        del_t_crit = del_t if del_t is not None else critical_timestep()
        self.sim_time = SimulationTime(dt=del_t_crit)

        self._law_on_submeshs = [
            create_law_on_submesh(law, local_cells, element_spaces, tangents=False)
            for law, local_cells in laws
        ]

        u_ = ufl.TestFunction(u.function_space)

        self.metadata = {"quadrature_degree": q_degree, "quadrature_scheme": "default"}
        self.dxm = ufl.dx(metadata=self.metadata)

        self.f_int_form = df.fem.form(
            ufl.inner(ufl_mandel_strain(u_, constraint), self.stress.current)
            * self.dxm,
            jit_options=jit_options,
            form_compiler_options=form_compiler_options,
        )

        self.f_ext_form = (
            df.fem.form(
                external_forces,
                jit_options=jit_options,
                form_compiler_options=form_compiler_options,
            )
            if external_forces is not None
            else None
        )
        self.f = u.copy()
        self.a = u.copy()
        self._bcs = bcs
        self._form_compiler_options = form_compiler_options
        self._jit_options = jit_options

        self.incr_disp = IncrementalDisplacement(u, q_degree)

    @df.common.timed("constitutive-form-evaluation-corotational")
    def step(self) -> None:
        # self._move_mesh_previous_to_midpoint()

        f_temp = np.zeros_like(self.f.x.array)
        df.fem.assemble_vector(f_temp, self.f_int_form)

        for law in self._law_on_submeshs:
            law.evaluate(self.sim_time, self.incr_disp, self.stress, self.tangent)

        # self._move_mesh_midpoint_to_final()

        self.stress.scatter_current()
        self.tangent.x.scatter_forward()


def _critical_timestep(
    laws: list[tuple[IncrSmallStrainModel, np.ndarray]],
    density: list[float],
    u: df.fem.Function,
    method: str = "cdm",
) -> np.ndarray:
    """
    Determines the critical timesteps for all submeshes. This assumes that the constitutive law
    returns a linear elastic tangent for $\sigma=0,\varepsilon=0$. The input is not verified for 
    consistency as this function is supposed to be called in the CDMSolver or any other solver.
    """
    method_to_factor = {"cdm": 2}
    factor = method_to_factor[method]
    mesh = u.function_space.mesh
    cell_type = mesh.ufl_cell().cellname()
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
        law.evaluate(0.0, 1e-12, grad_del_u, stress, tangent, history)
        assert np.linalg.norm(tangent) > 0.0, (
            "The constitutive law must return a non-zero tangent"
        )
        h = mesh.h(mesh.topology.dim, cells)
        h_min = h.min()
        omega = _max_frequency_one_element(h_min, u, tangent, density_)
        del_t.append(factor / omega)
    return np.array(del_t)


def _max_frequency_one_element(
    h: float, u: df.fem.Function, tangent: np.ndarray, density: float
) -> float:
    mesh_cell = u.function_space.mesh.ufl_cell().cellname()

    V_degree = u.function_space.ufl_element().degree()

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
    V_h = df.fem.functionspace(h_mesh, ("CG", V_degree, h_mesh.geometry.dim))
    h_u, h_v = ufl.TrialFunction(V_h), ufl.TestFunction(V_h)
    tangent_ufl = ufl.as_matrix(tangent.tolist())

    K_form = df.fem.form(
        ufl.inner(ufl.dot(tangent_ufl, ufl_mandel_strain(h_u)), ufl_mandel_strain(h_v))
        * ufl.dx
    )
    M_form = df.fem.form(density * ufl.inner(h_u, h_v) * ufl.dx)

    h_K, h_M = (
        df.fem.assemble_matrix(K_form),
        df.fem.assemble_matrix(M_form),
    )
    h_K.scatter_reverse()
    h_M.scatter_reverse()
    h_M = h_M.to_dense()
    h_K = h_K.to_dense()
    max_eig = np.linalg.norm(eigvals(h_K, h_M), np.inf)
    return max_eig**0.5
