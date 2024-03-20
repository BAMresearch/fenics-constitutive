from __future__ import annotations

import basix
import dolfinx as df
import numpy as np
import ufl
from petsc4py import PETSc

from .interfaces import IncrSmallStrainModel
from .maps import SubSpaceMap, build_subspace_map
from .stress_strain import ufl_mandel_strain


def build_history(
    law: IncrSmallStrainModel, mesh: df.mesh.Mesh, q_degree: int
) -> df.fem.Function | dict[str, df.fem.Function] | None:
    """Build the history space and function(s) for the given law.
    
    Args:
        law: The constitutive law.
        mesh: Either the full mesh for a homogenous domain or the submesh.
        q_degree: The quadrature degree.
    
    Returns:
        The history function(s) for the given law.

    """
    match law.history_dim:
        case int():
            Qh = ufl.VectorElement(
                "Quadrature",
                mesh.ufl_cell(),
                q_degree,
                quad_scheme="default",
                dim=law.history_dim,
            )
            history_space = df.fem.FunctionSpace(mesh, Qh)
            history = df.fem.Function(history_space)
        case None:
            history = None
        case dict():
            history = {}
            for key, value in law.history_dim.items():
                if isinstance(value, int):
                    Qh = ufl.VectorElement(
                        "Quadrature",
                        mesh.ufl_cell(),
                        q_degree,
                        quad_scheme="default",
                        dim=value,
                    )
                elif isinstance(value, tuple):
                    Qh = ufl.TensorElement(
                        "Quadrature",
                        mesh.ufl_cell(),
                        q_degree,
                        quad_scheme="default",
                        shape=value,
                    )
                history_space = df.fem.FunctionSpace(mesh, Qh)
                history[key] = df.fem.Function(history_space)
    return history


class IncrSmallStrainProblem(df.fem.petsc.NonlinearProblem):
    def __init__(
        self,
        laws: list[tuple[IncrSmallStrainModel, np.ndarray]],
        u: df.fem.Function,
        bcs: list[df.fem.DirichletBCMetaClass],
        q_degree: int = 1,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
    ):
        mesh = u.function_space.mesh
        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        cells = np.arange(0, num_cells, dtype=np.int32)

        constraint = laws[0][0].constraint
        assert all(
            law[0].constraint == constraint for law in laws
        ), "All laws must have the same constraint"

        gdim = mesh.ufl_cell().geometric_dimension()
        assert constraint.geometric_dim() == gdim, "Geometric dimension mismatch between mesh and laws"

        QVe = ufl.VectorElement(
            "Quadrature",
            mesh.ufl_cell(),
            q_degree,
            quad_scheme="default",
            dim=constraint.stress_strain_dim(),
        )
        QTe = ufl.TensorElement(
            "Quadrature",
            mesh.ufl_cell(),
            q_degree,
            quad_scheme="default",
            shape=(constraint.stress_strain_dim(), constraint.stress_strain_dim()),
        )
        QV = df.fem.FunctionSpace(mesh, QVe)
        QT = df.fem.FunctionSpace(mesh, QTe)

        self.laws: list[tuple[IncrSmallStrainModel, np.ndarray]] = []
        self.submesh_maps: list[SubSpaceMap] = []

        self._del_strain = []
        self._stress = []
        self._history_0 = []
        self._history_1 = []
        self._tangent = []

        self._time = 0.0  # time at the end of the increment

        with df.common.Timer("submeshes-and-data-structures"):
            if len(laws) > 1:
                for law, cells in laws:
                    self.laws.append((law, cells))

                    # ### submesh and subspace for strain, stress
                    subspace_map, submesh, QV_subspace = build_subspace_map(
                        cells, QV, return_subspace=True
                    )
                    self.submesh_maps.append(subspace_map)
                    self._del_strain.append(df.fem.Function(QV_subspace))
                    self._stress.append(df.fem.Function(QV_subspace))

                    #subspace for tanget
                    QT_subspace = df.fem.FunctionSpace(submesh, QTe)
                    self._tangent.append(df.fem.Function(QT_subspace))

                    #subspaces for history
                    history_0 = build_history(law, submesh, q_degree)
                    history_1 = (
                        {key: fn.copy() for key, fn in history_0.items()}
                        if isinstance(history_0, dict)
                        else history_0
                    )
                    self._history_0.append(history_0)
                    self._history_1.append(history_1)
            else:
                law, cells = laws[0]
                self.laws.append((law, cells))
                
                self._del_strain.append(df.fem.Function(QV))

                #Spaces for history
                history_0 = build_history(law, mesh, q_degree)
                history_1 = (
                    {key: fn.copy() for key, fn in history_0.items()}
                    if isinstance(history_0, dict)
                    else history_0
                )
                self._history_0.append(history_0)
                self._history_1.append(history_1)

        self.stress_0 = df.fem.Function(QV)
        self.stress_1 = df.fem.Function(QV)
        self.tangent = df.fem.Function(QT)

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

        self.del_strain_expr = df.fem.Expression(
            ufl_mandel_strain(self._u - self._u0, constraint), self.q_points
        )

        # ### Note on JIT compilation of UFL forms
        # if super().__init__(R, u, bcs, dR) is called within ElasticityProblem.__init__
        # the user cannot add Neumann BCs.
        # Therefore, the compilation (i.e. call to super().__init__()) is done when
        # df.nls.petsc.NewtonSolver is initialized.
        # df.nls.petsc.NewtonSolver will call
        # self._A = fem.petsc.create_matrix(problem.a) and hence it would suffice
        # to override the property ``a`` of NonlinearProblem.

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
                form_compiler_options=self._form_compiler_options if self._form_compiler_options is not None else {},
                jit_options=self._jit_options if self._jit_options is not None else {},
            )

        return self._a

    def form(self, x: PETSc.Vec):
        """This function is called before the residual or Jacobian is
        computed. This is usually used to update ghost values.
        Parameters
        ----------
        x
            The vector containing the latest solution
        """
        super().form(x)

        assert (
            x is self._u.vector
        ), "The solution vector must be the same as the one passed to the MechanicsProblem"
        if len(self.laws) > 1:
            for k, (law, cells) in enumerate(self.laws):
                with df.common.Timer("strain_evaluation"):
                    #TODO: test this!!
                    self.del_strain_expr.eval(cells, self._del_strain[k].x.array.reshape(cells.size, -1))

                with df.common.Timer("stress_evaluation"):
                    self.submesh_maps[k].map_to_child(self.stress_0, self._stress[k])
                    self._history_1[k].x.array[:] = self._history_0[k].x.array
                    law.evaluate(
                        self._time,
                        self._del_strain[k].x.array,
                        self._stress[k].x.array,
                        self._tangent[k].x.array,
                        self._history_1[k].x.array,  # history,
                    )

                with df.common.Timer("stress-local-to-global"):
                    self.submesh_maps[k].map_to_parent(self._stress[k], self.stress_1)
                    self.submesh_maps[k].map_to_parent(self._tangent[k], self.tangent)
        else:
            law, cells = self.laws[0]
            with df.common.Timer("strain_evaluation"):
                self.del_strain_expr.eval(cells, self._del_strain[0].x.array)

            with df.common.Timer("stress_evaluation"):
                self.stress_1.x.array[:] = self.stress_0.x.array
                self._history_1[0].x.array[:] = self._history_0[0].x.array
                law.evaluate(
                    self._time,
                    self._del_strain[0].x.array,
                    self.stress_1.x.array,
                    self.tangent.x.array,
                    self._history_1[0].x.array,  # history,
                )

        self.stress_1.x.scatter_forward()
        self.tangent.x.scatter_forward()

    def update(self):
        # update the current displacement, stress and history
        # works for both homogeneous and inhomogeneous domains
        self._u0.x.array[:] = self._u.x.array
        self._u0.x.scatter_forward()

        self.stress_0.x.array[:] = self.stress_1.x.array
        self.stress_0.x.scatter_forward()
        
        for k, (law, _) in enumerate(self.laws):
            match law.history_dim:
                case int():
                    self._history_0[k].x.array[:] = self._history_1[k].x.array
                    self._history_0[k].x.scatter_forward()
                case None:
                    pass
                case dict():
                    for key in law.history_dim:
                        self._history_0[k][key].x.array[:] = self._history_1[k][key].x.array
                        self._history_0[k][key].x.scatter_forward()
