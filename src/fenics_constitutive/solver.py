from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import basix
import basix.ufl
import dolfinx as df
import numpy as np
import ufl
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc

from .interfaces import IncrSmallStrainModel
from .maps import SubSpaceMap, build_subspace_map
from .stress_strain import ufl_mandel_strain


def build_history(
    law: IncrSmallStrainModel, mesh: df.mesh.Mesh, q_degree: int
) -> dict[str, df.fem.Function] | None:
    """Build the history space and function(s) for the given law.

    Args:
        law: The constitutive law.
        mesh: Either the full mesh for a homogenous domain or the submesh.
        q_degree: The quadrature degree.

    Returns:
        The history function(s) for the given law or None if the history dimension is 0.

    """
    if law.history_dim is None:
        return None

    history = {}
    for key, value in law.history_dim.items():
        value_shape = (value,) if isinstance(value, int) else value
        Q = basix.ufl.quadrature_element(
            mesh.topology.cell_name(), value_shape=value_shape, degree=q_degree
        )
        history_space = df.fem.functionspace(mesh, Q)
        history[key] = df.fem.Function(history_space)
    return history


class IncrSmallStrainProblem(NonlinearProblem):
    """
    A nonlinear problem for incremental small strain models. To be used with
    the dolfinx NewtonSolver.

    Args:
        laws: A list of tuples where the first element is the constitutive law and the second
            element is the cells for the submesh. If only one law is provided, it is assumed
            that the domain is homogenous.
        u: The displacement field. This is the unknown in the nonlinear problem.
        bcs: The Dirichlet boundary conditions.
        q_degree: The quadrature degree (Polynomial degree which the quadrature rule needs to integrate exactly).
        del_t: The time increment.
        form_compiler_options: The options for the form compiler.
        jit_options: The options for the JIT compiler.

    Note:
        If `super().__init__(R, u, bcs, dR)` is called within the __init__ method,
        the user cannot add Neumann BCs. Therefore, the compilation (i.e. call to
        `super().__init__()`) is done when `df.nls.petsc.NewtonSolver` is initialized.
        The solver will call `self._A = fem.petsc.create_matrix(problem.a)` and hence
        we override the property ``a`` of NonlinearProblem to ensure that the form is compiled.
    """

    def __init__(
        self,
        laws: list[tuple[IncrSmallStrainModel, np.ndarray]] | IncrSmallStrainModel,
        u: df.fem.Function,
        bcs: list[df.fem.DirichletBC],
        q_degree: int,
        del_t: float = 1.0,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
    ):
        mesh = u.function_space.mesh
        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        cells = np.arange(0, num_cells, dtype=np.int32)
        if isinstance(laws, IncrSmallStrainModel):
            laws = [(laws, cells)]

        constraint = laws[0][0].constraint
        assert all(law[0].constraint == constraint for law in laws), (
            "All laws must have the same constraint"
        )

        gdim = mesh.geometry.dim
        assert constraint.geometric_dim == gdim, (
            "Geometric dimension mismatch between mesh and laws"
        )

        QVe = basix.ufl.quadrature_element(
            mesh.topology.cell_name(),
            value_shape=(constraint.stress_strain_dim,),
            degree=q_degree,
        )
        QTe = basix.ufl.quadrature_element(
            mesh.topology.cell_name(),
            value_shape=(
                constraint.stress_strain_dim,
                constraint.stress_strain_dim,
            ),
            degree=q_degree,
        )
        Q_grad_u_e = basix.ufl.quadrature_element(
            mesh.topology.cell_name(), value_shape=(gdim, gdim), degree=q_degree
        )
        QV = df.fem.functionspace(mesh, QVe)
        QT = df.fem.functionspace(mesh, QTe)

        self.laws = laws
        # self.submesh_maps: list[SubSpaceMap] = []

        self._del_t = del_t  # time increment
        self._time = 0  # global time will be updated in the update method

        self.stress_0 = df.fem.Function(QV)
        self.stress_1 = df.fem.Function(QV)
        self.tangent = df.fem.Function(QT)

        self.quadrature_data = QuadratureData(
            laws, u, self.stress_1, q_degree, QVe, Q_grad_u_e, QTe, True
        )

        u_, du = ufl.TestFunction(u.function_space), ufl.TrialFunction(u.function_space)

        self.metadata = {"quadrature_degree": q_degree, "quadrature_scheme": "default"}
        self.dxm = ufl.dx(metadata=self.metadata)

        self.R_form = (
            ufl.inner(ufl_mandel_strain(u_, constraint), self.stress_1) * self.dxm
        )
        self.dR_form = (
            ufl.inner(
                ufl_mandel_strain(du, constraint),
                ufl.dot(self.tangent, ufl_mandel_strain(u_, constraint)),
            )
            * self.dxm
        )

        self._u = u
        self._u0 = u.copy()
        self._bcs = bcs
        self._form_compiler_options = form_compiler_options
        self._jit_options = jit_options

        basix_celltype = getattr(basix.CellType, mesh.topology.cell_type.name)
        self.q_points, _ = basix.make_quadrature(basix_celltype, q_degree)

        self.del_grad_u_expr = df.fem.Expression(
            ufl.nabla_grad(self._u - self._u0), self.q_points
        )

    @property
    def a(self) -> df.fem.FormMetaClass:
        """Compiled bilinear form (the Jacobian form)"""

        if not hasattr(self, "_a"):
            # ensure compilation of UFL forms
            super().__init__(
                self.R_form,
                self._u,
                self._bcs,
                self.dR_form,
                form_compiler_options=self._form_compiler_options
                if self._form_compiler_options is not None
                else {},
                jit_options=self._jit_options if self._jit_options is not None else {},
            )

        return self._a

    @df.common.timed("constitutive-form-evaluation")
    def form(self, x: PETSc.Vec) -> None:
        """This function is called before the residual or Jacobian is
        computed. This is usually used to update ghost values, but here
        we use it to update the stress, tangent and history.

        Args:
            x: The vector containing the latest solution

        """
        super().form(x)
        # This assertion can fail, even if everything is correct.
        # Left here, because I would like the check to work someday again.
        # assert (
        #    x.array.data == self._u.vector.array.data
        # ), f"The solution vector must be the same as the one passed to the MechanicsProblem. Got {x.array.data} and {self._u.vector.array.data}"

        # this copies the data from the vector x to the function _u
        x.copy(self._u.x.petsc_vec)
        self._u.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        evaluate_model(
            self.quadrature_data,
            self.laws,
            self.del_grad_u_expr,
            self._time,
            self._del_t,
            self.stress_1,
            self.stress_0,
            self.tangent,
        )

    def update(self) -> None:
        """
        Update the current displacement, stress and history.
        """
        self._u0.x.array[:] = self._u.x.array
        self._u0.x.scatter_forward()

        self.stress_0.x.array[:] = self.stress_1.x.array
        self.stress_0.x.scatter_forward()

        for k, (law, _) in enumerate(self.laws):
            # law.update()
            if (
                law.history_dim is not None
                and self.quadrature_data.history_initial is not None
            ):
                for key in law.history_dim:
                    self.quadrature_data.history_initial[k][key].x.array[:] = (
                        self.quadrature_data.history[k][key].x.array
                    )
                    self.quadrature_data.history_initial[k][key].x.scatter_forward()

        # time update
        self._time += self._del_t


@dataclass
class QuadratureData:
    """
    A data handler for the constitutive law. This is used to store the data
    for the constitutive law and to update it.

    Args:
        laws: A list of tuples where the first element is the constitutive law and the second
            element is the cells for the submesh. If only one law is provided, it is assumed
            that the domain is homogenous.
        u: The displacement field. This is the unknown in the nonlinear problem.
        q_degree: The quadrature degree (Polynomial degree which the quadrature rule needs to integrate exactly).
    """

    history_initial: list[dict[str, df.fem.Function]] | None
    history: list[dict[str, df.fem.Function]] | None
    stress: list[df.fem.Function] | None
    tangent: list[df.fem.Function] | None
    del_grad_u: list[df.fem.Function]
    supspace_maps: list[SubSpaceMap] | None

    def __init__(
        self,
        laws: list[tuple[IncrSmallStrainModel, np.ndarray]],
        u: df.fem.Function,
        stress: df.fem.Function,
        q_degree: int,
        stress_element: basix.ufl._ElementBase,
        del_grad_u_element: basix.ufl._ElementBase,
        tangent_element: basix.ufl._ElementBase | None,
        use_initial_state: bool = True,
    ) -> None:
        mesh = u.function_space.mesh
        _submesh_maps = []
        _stress = [] if len(laws) > 1 else None
        _del_grad_u = []
        _tangent = [] if tangent_element is not None else None
        _history = []
        _history_initial = [] if use_initial_state else None
        for law, cells in laws:
            # default case for homogenous domain
            submesh = mesh

            if len(laws) > 1:
                # ### submesh and subspace for strain, stress
                subspace_map, submesh, QV_subspace = build_subspace_map(
                    cells, stress.function_space, return_subspace=True
                )
                _submesh_maps.append(subspace_map)
                _stress.append(df.fem.Function(QV_subspace))

            # subspace for grad u
            Q_grad_u_subspace = df.fem.functionspace(submesh, del_grad_u_element)
            _del_grad_u.append(df.fem.Function(Q_grad_u_subspace))

            # subspace for tanget
            if _tangent is not None:
                QT_subspace = df.fem.functionspace(submesh, tangent_element)
                _tangent.append(df.fem.Function(QT_subspace))

            # subspaces for history
            history = build_history(law, submesh, q_degree)
            _history.append(history)
            if use_initial_state:
                history_initial = (
                    {key: fn.copy() for key, fn in history.items()}
                    if isinstance(history, dict)
                    else history
                )
                _history_initial.append(history_initial)

        self.history_initial = _history_initial
        self.history = _history
        self.stress = _stress
        self.tangent = _tangent
        self.del_grad_u = _del_grad_u
        self.supspace_maps = _submesh_maps


def evaluate_model(
    submesh_data: QuadratureData,
    laws: list[tuple[IncrSmallStrainModel, np.ndarray]],
    grad_del_u_expr: df.fem.Expression,
    time: float,
    del_t: float,
    stress: df.fem.Function,
    stress_initial: df.fem.Function | None,
    tangent: df.fem.Function | None,
) -> None:
    """Evaluate the constitutive law for the given quadrature data.
    This function does not depend on PETSc or the nonlinear solver and can therefore
    be used in a solver that uses scipy or explicit solvers that only work on the numpy
    level. It updates the stress, tangent and history functions based on the laws provided.

    Args:
        submesh_data: The submesh data containing the history, stress, tangent and del_grad_u.
        laws: The list of constitutive laws.
        grad_del_u_expr: The expression for the gradient of the displacement.
        time: The current time.
        del_t: The time increment.
        stress: The stress function to be updated.
        stress_initial: The initial stress function (if applicable).
        tangent: The tangent function to be updated.

    """
    for k, (law, cells) in enumerate(laws):
        submesh_data.del_grad_u[k].interpolate(
            grad_del_u_expr,
            cells0=cells.copy(),
            cells1=np.arange(cells.size, dtype=np.int32),
        )
        submesh_data.del_grad_u[k].x.scatter_forward()

        if len(laws) > 1:
            submesh_data.supspace_maps[k].map_to_child(
                stress_initial, submesh_data.stress[k]
            )
            stress_input = submesh_data.stress[k].x.array
            tangent_input = submesh_data.tangent[k].x.array if tangent else None
        else:
            stress.x.array[:] = stress_initial.x.array
            stress.x.scatter_forward()
            stress_input = stress.x.array
            tangent_input = tangent.x.array if tangent else None

        history_input = None
        if law.history_dim is not None and submesh_data.history_initial is not None:
            history_input = {}
            for key in law.history_dim:
                # copy initial history values to current history
                submesh_data.history[k][key].x.array[:] = submesh_data.history_initial[
                    k
                ][key].x.array
                history_input[key] = submesh_data.history[k][key].x.array
        elif law.history_dim is not None and submesh_data.history_initial is None:
            history_input = {}
            for key in law.history_dim:
                # directly overwrite values
                history_input[key] = submesh_data.history[k][key].x.array

        with df.common.Timer("constitutive-law-evaluation"):
            law.evaluate(
                time,
                del_t,
                submesh_data.del_grad_u[k].x.array,
                stress_input,
                tangent_input,
                history_input,
            )

        if len(laws) > 1:
            submesh_data.supspace_maps[k].map_to_parent(submesh_data.stress[k], stress)
            if tangent is not None:
                submesh_data.supspace_maps[k].map_to_parent(
                    submesh_data.tangent[k], tangent
                )

    stress.x.scatter_forward()
    if tangent is not None:
        tangent.x.scatter_forward()


@dataclass
class DynamicSolver(ABC):
    """
    An abstract class for a dynamic solver (Either explicit or implicit).
    """

    laws: list[tuple[IncrSmallStrainModel, np.ndarray]]
    u: df.fem.Function
    v: df.fem.Function
    f: df.fem.Function
    f_form: df.fem.Form
    del_grad_u_expr: df.fem.Expression
    bcs: list[df.fem.DirichletBC]
    del_t_min: float = 1.0
    del_t_max: float = 1.0
    quadrature_data: QuadratureData
    form_compiler_options: dict | None = None
    jit_options: dict | None = None

    @abstractmethod
    def __init__(
        self,
        laws: list[tuple[IncrSmallStrainModel, np.ndarray]] | IncrSmallStrainModel,
        u: df.fem.Function,
        v: df.fem.Function,
        bcs: list[df.fem.DirichletBC],
        q_degree: int,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
    ) -> None:
        pass

    @abstractmethod
    def step(self) -> None:
        """
        Perform a single time step using the dynamic solver.
        """

    @abstractmethod
    def set_timestep(self, del_t_min: float, del_t_max: float | None = None) -> None:
        """
        Set the minimum and maximum time step for the dynamic solver.

        Args:
            del_t_min: The minimum time step.
            del_t_max: The maximum time step. If `None`, it is set to the same value as del_t_min.
        """


@dataclass(frozen=True)
class CentralDifferenceMethod(DynamicSolver):
    """
    A class for the Central Difference Method (CDM) solver for incremental small strain models.

    laws: A list of tuples where the first element is the constitutive law and the second
            element is the cells for the submesh. If only one law is provided, it is assumed
            that the domain is homogenous.
    u: The displacement field. This is the unknown in the nonlinear problem.
    bcs: The Dirichlet boundary conditions.
    q_degree: The quadrature degree (Polynomial degree which the quadrature rule needs to integrate exactly).
    del_t: The time increment.
    form_compiler_options: The options for the form compiler.
    jit_options: The options for the JIT compiler.
    """

    laws: list[tuple[IncrSmallStrainModel, np.ndarray]]
    u: df.fem.Function
    v: df.fem.Function
    f: df.fem.Function
    f_form: df.fem.Form
    del_grad_u_expr: df.fem.Expression
    bcs: list[df.fem.DirichletBC]
    del_t_min: float = 1.0
    del_t_max: float = 1.0
    del_t: df.fem.Constant
    quadrature_data: QuadratureData
    diagonal_mass: df.fem.Function
    form_compiler_options: dict | None = None
    jit_options: dict | None = None

    def __init__(
        self,
        laws: list[tuple[IncrSmallStrainModel, np.ndarray]] | IncrSmallStrainModel,
        u0: df.fem.Function,
        v0: df.fem.Function,
        bcs: list[df.fem.DirichletBC],
        q_degree: int,
        del_t_min: float,
        del_t_max: float | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
    ) -> None:
        mesh = u0.function_space.mesh
        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        if isinstance(laws, IncrSmallStrainModel):
            cells = np.arange(0, num_cells, dtype=np.int32)
            laws = [(laws, cells)]

        constraint = laws[0][0].constraint
        assert all(law[0].constraint == constraint for law in laws), (
            "All laws must have the same constraint"
        )

        gdim = mesh.geometry.dim
        assert constraint.geometric_dim == gdim, (
            "Geometric dimension mismatch between mesh and laws"
        )

        stress_element = basix.ufl.quadrature_element(
            mesh.topology.cell_name(),
            value_shape=(constraint.stress_strain_dim,),
            degree=q_degree,
        )
        grad_v_element = basix.ufl.quadrature_element(
            mesh.topology.cell_name(), value_shape=(gdim, gdim), degree=q_degree
        )
        Q_stress = df.fem.functionspace(mesh, stress_element)

        self.laws = laws
        # self.submesh_maps: list[SubSpaceMap] = []

        self.stress = df.fem.Function(Q_stress)

        self.quadrature_data = QuadratureData(
            laws,
            u0,
            self.stress,
            q_degree,
            stress_element,
            grad_v_element,
            None,
            False,
        )

        u_ = ufl.TestFunction(u0.function_space)

        self.metadata = {"quadrature_degree": q_degree, "quadrature_scheme": "default"}
        self.dxm = ufl.dx(metadata=self.metadata)

        self.R_form = (
            ufl.inner(ufl_mandel_strain(u_, constraint), self.stress) * self.dxm
        )

        self.u = u0

        self.bcs = bcs
        self.form_compiler_options = form_compiler_options
        self.jit_options = jit_options

        basix_celltype = getattr(basix.CellType, mesh.topology.cell_type.name)
        self.q_points, _ = basix.make_quadrature(basix_celltype, q_degree)
        self.del_t_min = del_t_min
        self.del_t_max = del_t_max if del_t_max is not None else del_t_min
        self.del_t = df.fem.Constant(mesh, dtype=np.float64, value=del_t_min)
        self.del_grad_u_expr = df.fem.Expression(
            self.del_t * ufl.nabla_grad(self.v), self.q_points
        )

    def step(self) -> None:
        """
        Perform a single time step using the Central Difference Method (CDM).
        This method updates the displacement and velocity fields based on the laws
        and the time increment.
        """

        with self.["f"].vector.localForm() as f_local:
            f_local.set(0.0)´
        df.fem.petsc.assemble_vector(self.fields["f"].vector, self.f_int_form)
        self.fields["f"].x.scatter_reverse(ScatterMode.add)

        if self.external_forces is not None:
            external_forces = self.external_forces(self.t)
            self.fields["f"].vector.array[:] += external_forces.vector.array
            self.fields["f"].x.scatter_forward()

        self.fields["v"].vector.array[:] += del_t_mid * self.M.vector.array * self.fields["f"].vector.array
        self.fields["v"].x.scatter_forward()

        df.fem.set_bc(self.fields["v"].vector, self.bcs)
        # ghost entries are needed
        self.fields["v"].x.scatter_forward()
        # use v.x instead of v.vector, since mesh update requires ghost entries

        du_half = (0.5 * self.del_t) * self.fields["v"].x.array

        set_mesh_coordinates(self.function_space.mesh, du_half, mode="add")

        # basically, evaluate the nonlocal variable here
        if intermediate_step is not None:
            intermediate_step(h)

        self.stress_update(self.del_t)

        self.fields["u"].x.array[:] += 2.0 * du_half
        self.fields["u"].x.scatter_forward()

        set_mesh_coordinates(self.function_space.mesh, du_half, mode="add")

        if self.total_energy is not None:
            external_forces_0 = external_forces.vector.array.copy()

            # undo mesh update
            # set_mesh_coordinates(self.function_space.mesh, -du_half, mode="add")
            external_forces = self.external_forces(self.t + self.del_t)

            external_forces.vector.array[:] = 0.5 * (external_forces_0 + external_forces.vector.array)

            # energy_form = df.fem.form(ufl.inner(external_forces, self.fields["v"]) * ufl.ds)
            # energy_increment = df.fem.assemble_scalar(energy_form)
            energy_increment = np.inner(external_forces.vector.array, self.fields["v"].vector.array)
            self.total_energy += self.del_t * energy_increment
            # redo mesh update
            # set_mesh_coordinates(self.function_space.mesh, du_half, mode="add")
        #
        self.t += self.del_t

    def set_timestep(self, del_t_min, del_t_max=None) -> None:
        self.del_t_min = del_t_min
        self.del_t_max = del_t_max if del_t_max is not None else del_t_min
