from dataclasses import dataclass
from typing import Any
import basix
import dolfinx as df
import numpy as np
import ufl
from petsc4py import PETSc
from .interfaces import IncrSmallStrainModel
from .stress_strain import ufl_mandel_strain

from .maps import SubSpaceMap, build_subspace_map

def build_history(
    law: IncrSmallStrainModel, submesh: df.mesh.Mesh, q_degree: int
) -> df.fem.Function | dict[str, df.fem.Function] | None:
    match law.history_dim:
        case int():
            Qh = ufl.VectorElement(
                "Quadrature",
                submesh.ufl_cell(),
                q_degree,
                quad_scheme="default",
                dim=law.history_dim,
            )
            history_space = df.fem.FunctionSpace(submesh, Qh)
            history = df.fem.Function(history_space)
        case None:
            history = None
        case dict():
            history = {}
            for key, value in law.history_dim.items():
                if isinstance(value, int):
                    Qh = ufl.VectorElement(
                        "Quadrature",
                        submesh.ufl_cell(),
                        q_degree,
                        quad_scheme="default",
                        dim=value,
                    )
                elif isinstance(value, tuple):
                    Qh = ufl.TensorElement(
                        "Quadrature",
                        submesh.ufl_cell(),
                        q_degree,
                        quad_scheme="default",
                        shape=value,
                    )
                history_space = df.fem.FunctionSpace(submesh, Qh)
                history[key] = df.fem.Function(history_space)
    return history



class IncrMechanicsProblem(df.fem.petsc.NonlinearProblem):
    def __init__(
        self,
        # laws = [(law_1, cells_1), (law_2, cells_2), ...]
        laws: list[tuple[IncrSmallStrainModel, np.ndarray]],
        # u0 is basically a history variable
        u0: df.fem.Function,
        bcs: list[df.fem.DirichletBCMetaClass],
        q_degree: int = 1,
        form_compiler_options: dict = {},
        jit_options: dict = {},
    ):
        mesh = u0.function_space.mesh
        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        self.cells = np.arange(0, num_cells, dtype=np.int32)

        self.gdim = mesh.ufl_cell().geometric_dimension()

        constraint = laws[0][0].constraint
        assert all(
            law[0].constraint == constraint for law in laws
        ), "All laws must have the same constraint"

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

        self.laws = []
        # TODO: We can probably reduce to one map per submesh
        self.submesh_maps = []
        #self.QV_views = []
        #self.Qh_views = []
        #self.QT_views = []

        self._strain = []
        self._stress = []
        self._history_0 = []
        self._history_1 = []
        self._tangent = []

        self._time = 0.0  # time at the end of the increment

        with df.common.Timer("submeshes-and-data-structures"):
            for law, cells in laws:
                self.laws.append((law, cells))
                

                # ### submesh and subspace for strain, stress
                subspace_map, submesh, QV_subspace = build_subspace_map(cells, QV, return_subspace=True)
                self.submesh_maps.append(subspace_map)
                self._strain.append(df.fem.Function(QV_subspace))
                self._stress.append(df.fem.Function(QV_subspace))

                # ### submesh and subspace for tanget
                QT_subspace = df.fem.FunctionSpace(submesh, QTe)
                self._tangent.append(df.fem.Function(QT_subspace))

                # ### submesh and subspace for history
                history_0, Qh_views = build_history(law, submesh, q_degree)
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

        u_, du = ufl.TestFunction(u0.function_space), ufl.TrialFunction(u0.function_space)

        self.metadata = {"quadrature_degree": q_degree, "quadrature_scheme": "default"}
        self.dxm = ufl.dx(metadata=self.metadata)

        self.R_form = ufl.inner(ufl_mandel_strain(u_, constraint), self.stress) * self.dxm
        self.dR_form = ufl.inner(ufl_mandel_strain(du, constraint), ufl.dot(self.tangent, ufl_mandel_strain(u_, constraint))) * self.dxm

        self._u0 = u0
        self._bcs = bcs
        self._form_compiler_options = form_compiler_options
        self._jit_options = jit_options

        basix_celltype = getattr(basix.CellType, mesh.topology.cell_type.name)
        self.q_points, _ = basix.make_quadrature(basix_celltype, q_degree)

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
                form_compiler_options=self._form_compiler_options,
                jit_options=self._jit_options,
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
        #TODO: look if the creation of the expression slows things down
        strain_expr = df.fem.Expression(ufl_mandel_strain(x - self._u0, self.constraint), self.q_points)

        for k, (law, cells) in enumerate(self.laws):
            with df.common.Timer("strain_evaluation"):
                strain_expr.eval(
                    cells, self._strain[k].x.array.reshape(cells.size, -1)
                )

            with df.common.Timer("stress_evaluation"):
                self.submesh_maps[k].map_to_child(self.stress_0, self._stress[k])
                self._history_1[k].x.array[:] = self._history_0[k].x.array
                law.evaluate(
                    self._time,
                    self._strain[k].x.array,
                    self._stress[k].x.array,
                    self._tangent[k].x.array,
                    self._history_1[k].x.array,  # history,
                )


            with df.common.Timer("stress-local-to-global"):
                self.submesh_maps[k].map_to_parent(self._stress[k], self.stress_1)
                self.submesh_maps[k].map_to_parent(self._tangent[k], self.tangent)
            
        self.stress.x.scatter_forward()
        self.tangent.x.scatter_forward()

    def update(self):
        pass

