from typing import Union
from abc import ABC, abstractmethod, abstractproperty
from enum import Enum

import numpy as np
import dolfinx as df
import ufl
import basix
from petsc4py import PETSc

__all__ = [
    "Constraint",
    "IncrSmallStrainModel",
    "IncrSmallStrainProblem",
]


class Constraint(Enum):
    """
    Enum for the model constraint.

    The constraint can either be:
    - UNIAXIAL_STRAIN
    - UNIAXIAL_STRESS
    - PLANE_STRAIN
    - PLANE_STRESS
    - FULL

    """

    UNIAXIAL_STRAIN = 1
    UNIAXIAL_STRESS = 2
    PLANE_STRAIN = 3
    PLANE_STRESS = 4
    FULL = 5

    def stress_strain_dim(self) -> int:
        """
        The stress-strain dimension of the constraint.

        Returns:
            The stress-strain dimension.
        """
        match self:
            case Constraint.UNIAXIAL_STRAIN:
                return 1
            case Constraint.UNIAXIAL_STRESS:
                return 1
            case Constraint.PLANE_STRAIN:
                return 4
            case Constraint.PLANE_STRESS:
                return 4
            case Constraint.FULL:
                return 6

    def geometric_dim(self) -> int:
        """
        The geometric dimension for the constraint.

        Returns:
            The geometric dimension.
        """
        match self:
            case Constraint.UNIAXIAL_STRAIN:
                return 1
            case Constraint.UNIAXIAL_STRESS:
                return 1
            case Constraint.PLANE_STRAIN:
                return 2
            case Constraint.PLANE_STRESS:
                return 2
            case Constraint.FULL:
                return 3


class IncrSmallStrainModel(ABC):
    """
    Interface for incremental small strain models.
    """

    @abstractmethod
    def evaluate(
        self,
        del_t: float,
        grad_del_u: np.ndarray,
        mandel_stress: np.ndarray,
        tangent: np.ndarray,
        history: np.ndarray | dict[str, np.ndarray],
    ) -> None:
        """Evaluate the constitutive model and overwrite the stress, tangent and history.

        Args:
            del_t : The time increment.
            grad_del_u : The gradient of the increment of the displacement field.
            mandel_stress : The Mandel stress.
            tangent : The tangent.
            history : The history variable(s).
        """
        pass

    @abstractmethod
    def update(self) -> None:
        """
        Called after the solver has converged to update anything that is not
        contained in the history variable(s).
        For example: The model could contain the current time which is not
        stored in the history, but needs to be updated after each evaluation.
        """
        pass

    @abstractproperty
    def constraint(self) -> Constraint:
        """
        The constraint that the model is implemented for.

        Returns:
            The constraint.
        """
        pass

    @property
    def stress_strain_dim(self) -> int:
        """
        The stress-strain dimension that the model is implemented for.

        Returns:
            The stress-strain dimension.
        """
        return self.constraint.stress_strain_dim()

    @property
    def geometric_dim(self) -> int:
        """
        The geometric dimension that the model is implemented for.

        Returns:
            The geometric dimension.
        """
        return self.constraint.geometric_dim()

    @abstractproperty
    def history_dim(self) -> int | dict[str, int | tuple[int, int]]:
        """
        The dimensions of history variable(s). This is needed to tell the solver which quadrature
        spaces or arrays to build. If all history variables are stored in a single
        array, then the dimension of the array is returned. If the history variables are stored
        in seperate arrays (or functions), then a dictionary is returned with the name of the
        history variable as key and the dimension of the history variable as value.

        Returns:
            The dimension of the history variable(s).
        """
        pass


def _build_view(cells, V):
    mesh = V.mesh
    submesh, cell_map, _, _ = df.mesh.create_submesh(mesh, mesh.topology.dim, cells)
    fe = V.ufl_element()
    V_sub = df.fem.FunctionSpace(submesh, fe)

    submesh = V_sub.mesh
    view_parent = []
    view_child = []

    num_sub_cells = submesh.topology.index_map(submesh.topology.dim).size_local
    for cell in range(num_sub_cells):
        view_child.append(V_sub.dofmap.cell_dofs(cell))
        view_parent.append(V.dofmap.cell_dofs(cell_map[cell]))
    if view_child:
        return (
            np.hstack(view_parent),
            np.hstack(view_child),
            V_sub,
        )
    else:
        # it may be that a process does not own any of the cells in the submesh
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            V_sub,
        )


class IncrSmallStrainProblem(df.fem.petsc.NonlinearProblem, ABC):
    def __init__(
        self,
        laws: list[tuple[IncrSmallStrainModel, Union[np.ndarray, None]]],
        u: df.fem.Function,
        q_degree: int = 2,
    ):
        mesh = u.function_space.mesh
        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        self.cells = np.arange(0, num_cells, dtype=np.int32)
        self.gdim = mesh.ufl_cell().geometric_dimension()

        # sanity check, maybe add some more?
        assert all([self.gdim == law[0].geometric_dim for law in laws])
        assert all([laws[0][0].constraint is law[0].constraint for law in laws])
        self.constraint = laws[0][0].constraint

        # build global quadrature spaces
        sdim = laws[0][0].stress_strain_dim
        QVe = ufl.VectorElement(
            "Quadrature", mesh.ufl_cell(), q_degree, quad_scheme="default", dim=sdim
        )
        QTe = ufl.TensorElement(
            "Quadrature",
            mesh.ufl_cell(),
            q_degree,
            quad_scheme="default",
            shape=(sdim, sdim),
        )
        QV = df.fem.FunctionSpace(mesh, QVe)
        QT = df.fem.FunctionSpace(mesh, QTe)
        self.stress = df.fem.Function(QV)
        self.tangent = df.fem.Function(QT)

        # build submesh and related data structures if necessary
        self.laws = []
        self._strain = []
        self._stress = []
        self._tangent = []

        if len(laws) < 2:
            self.homogeneous = True
            law = laws[0][0]
            self.laws.append((law, self.cells))
            self._strain.append(df.fem.Function(QV))
            self._stress.append(self.stress)
            self._tangent.append(self.tangent)
        else:
            self.homogeneous = False
            self.QV_views = []
            self.QT_views = []

            with df.common.Timer("submeshes-and-data-structures"):
                for law, cells in laws:
                    self.laws.append((law, cells))

                    # ### submesh and subspace for strain, stress
                    QV_parent, QV_child, QV_sub = _build_view(cells, QV)
                    self.QV_views.append((QV_parent, QV_child, QV_sub))
                    self._strain.append(df.fem.Function(QV_sub))
                    self._stress.append(df.fem.Function(QV_sub))

                    # ### submesh and subspace for tanget
                    QT_parent, QT_child, QT_sub = _build_view(cells, QT)
                    self.QT_views.append((QT_parent, QT_child, QT_sub))
                    self._tangent.append(df.fem.Function(QT_sub))

        u_, du = ufl.TestFunction(u.function_space), ufl.TrialFunction(u.function_space)

        self.metadata = {"quadrature_degree": q_degree, "quadrature_scheme": "default"}
        self.dxm = ufl.dx(metadata=self.metadata)

        eps = self.eps
        self.R_form = ufl.inner(eps(u_), self.stress) * self.dxm
        self.dR_form = ufl.inner(eps(du), ufl.dot(self.tangent, eps(u_))) * self.dxm
        self.u = u

        basix_celltype = getattr(basix.CellType, mesh.topology.cell_type.name)
        self.q_points, _ = basix.make_quadrature(basix_celltype, q_degree)
        self.strain_expr = df.fem.Expression(eps(u), self.q_points)

    def compile(self, bcs):
        # FIXME
        # handle compilation internally such that no explicit call
        # by user is necessary

        R = self.R_form
        u = self.u
        dR = self.dR_form
        super().__init__(R, u, bcs=bcs, J=dR)

    def form(self, x: PETSc.Vec):
        """This function is called before the residual or Jacobian is
        computed. This is usually used to update ghost values.
        Parameters
        ----------
        x
            The vector containing the latest solution
        """
        super().form(x)

        # FIXME
        # how is ``time`` passed to ``evaluate`` for time dependent problems?
        time = 0.0
        history = np.array([])

        for k, (law, cells) in enumerate(self.laws):
            with df.common.Timer("strain_evaluation"):
                self.strain_expr.eval(
                    cells, self._strain[k].x.array.reshape(cells.size, -1)
                )

            with df.common.Timer("stress_evaluation"):
                law.evaluate(
                    time,
                    self._strain[k].x.array,
                    self._stress[k].x.array,
                    self._tangent[k].x.array,
                    history,
                )

            if not self.homogeneous:
                with df.common.Timer("stress-local-to-global"):
                    sdim = law.stress_strain_dim
                    parent, child, _ = self.QV_views[k]
                    stress_global = self.stress.x.array.reshape(-1, sdim)
                    stress_local = self._stress[k].x.array.reshape(-1, sdim)
                    stress_global[parent] = stress_local[child]
                    c_parent, c_child, _ = self.QT_views[k]
                    tangent_global = self.tangent.x.array.reshape(-1, sdim ** 2)
                    tangent_local = self._tangent[k].x.array.reshape(-1, sdim ** 2)
                    tangent_global[c_parent] = tangent_local[c_child]

        self.stress.x.scatter_forward()
        self.tangent.x.scatter_forward()

    @abstractmethod
    def eps(self, u) -> ufl.core.expr.Expr:
        """UFL expression for strain measure"""
        pass
