"""
Implicit gradient-enhanced damage model in dolfinx
==================================================

For the theory and the constitutive model, please refer to the dolfin version :ref:`gdm-label`.

Here, you see the "quadrature" approach working in dolfinx without the need of `LocalProject`.
"""

from gdm_constitutive import *
from gdm_analytic import PeerlingsAnalytic


import dolfinx as df
from petsc4py import PETSc
from mpi4py import MPI
import ufl
import basix

import matplotlib.pyplot as plt


class GDMProblemX:
    def __init__(self, mesh, mat, deg=2, q_deg=4, fd_np=None):
        self.mesh, self.mat= mesh, mat

        Ed = ufl.VectorElement("CG", mesh.ufl_cell(), degree=deg)
        Ee = ufl.FiniteElement("CG", mesh.ufl_cell(), degree=deg)
        self.V = df.fem.FunctionSpace(mesh, (Ed * Ee))

        self.Vd, self.Ve = self.V.sub(0), self.V.sub(1)
        self.u = df.fem.Function(self.V)

        q = "Quadrature"
        cell = mesh.ufl_cell()
        voigt = 3
        QF = ufl.FiniteElement(q, cell, q_deg, quad_scheme="default")
        QV = ufl.VectorElement(q, cell, q_deg, quad_scheme="default", dim=voigt)
        QT = ufl.TensorElement(
            q, cell, q_deg, quad_scheme="default", shape=(voigt, voigt)
        )

        VQF, VQV, VQT = [df.fem.FunctionSpace(mesh, Q) for Q in [QF, QV, QT]]
        self.q_sigma = df.fem.Function(VQV)
        self.q_e = df.fem.Function(VQF)
        self.q_eeq = df.fem.Function(VQF)

        self.q_dsigma_deps = df.fem.Function(VQT)
        self.q_dsigma_de = df.fem.Function(VQV)
        self.q_deeq_deps = df.fem.Function(VQV)

        # define functions
        dd, de = ufl.TrialFunctions(self.V)
        d_, e_ = ufl.TestFunctions(self.V)

        self.d, self.e = ufl.split(self.u)

        # define form
        self.metadata = {"quadrature_degree": q_deg, "quadrature_scheme": "default"}
        self.dxm = ufl.dx(metadata=self.metadata)

        # prepare strain evaluation
        eps = mat.eps
        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        self.cells = np.arange(0, num_cells, dtype=np.int32)

        basix_celltype = getattr(basix.CellType, mesh.topology.cell_type.name)
        self.q_points, wts = basix.make_quadrature(basix_celltype, q_deg)
        self.strain_expr = df.fem.Expression(eps(self.d), self.q_points)
        self.e_expr = df.fem.Expression(self.e, self.q_points)

        self.evaluate_constitutive_law()

        self.bcs = None

        if fd_np is None:
            fd_np = lambda x: 1.0

        Vfd = df.fem.FunctionSpace(mesh, ("DG", 0))
        fd = df.fem.Function(Vfd)
        fd.x.array[:] = fd_np(Vfd.tabulate_dof_coordinates())

        R = fd * ufl.inner(eps(d_), self.q_sigma) * self.dxm
        R += e_ * (self.e - self.q_eeq) * self.dxm
        R += ufl.dot(ufl.grad(e_), mat.l ** 2 * ufl.grad(self.e)) * self.dxm

        dR = fd * ufl.inner(eps(d_), self.q_dsigma_deps * eps(dd)) * self.dxm
        dR += fd * ufl.inner(eps(d_), self.q_dsigma_de * de) * self.dxm
        dR += e_ * (de - ufl.dot(self.q_deeq_deps, eps(dd))) * self.dxm
        dR += ufl.dot(ufl.grad(e_), mat.l ** 2 * ufl.grad(de)) * self.dxm

        self.R, self.dR = df.fem.form(R), df.fem.form(dR)

        self.solver = None

    def evaluate_constitutive_law(self):
        with df.common.Timer("compute strains"):
            strain = self.strain_expr.eval(self.cells)
            e = self.e_expr.eval(self.cells)

        with df.common.Timer("evaluate constitutive law"):
            self.mat.evaluate(strain.flatten(), e.flatten())

        with df.common.Timer("assign q space"):
            self.q_sigma.x.array[:] = self.mat.sigma.flat
            self.q_dsigma_deps.x.array[:] = self.mat.dsigma_deps.flat
            self.q_dsigma_de.x.array[:] = self.mat.dsigma_de.flat
            self.q_eeq.x.array[:] = self.mat.eeq.flat
            self.q_deeq_deps.x.array[:] = self.mat.deeq.flat

    def update(self):
        e = self.e_expr.eval(self.cells)
        self.mat.update(e.flatten())

    def form(self, x: PETSc.Vec):
        """This function is called before the residual or Jacobian is
        computed. This is usually used to update ghost values.
        Parameters
        ----------
        x
            The vector containing the latest solution
        """
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        self.evaluate_constitutive_law()

    def J(self, x: PETSc.Vec, A: PETSc.Mat, P=None):
        """Assemble the Jacobian matrix."""
        A.zeroEntries()
        df.fem.petsc.assemble_matrix(A, self.dR, bcs=self.bcs)
        A.assemble()

    def F(self, x: PETSc.Vec, b: PETSc.Vec):
        """Assemble the residual F into the vector b."""

        # Reset the residual vector
        with b.localForm() as b_local:
            b_local.set(0.0)
        df.fem.petsc.assemble_vector(b, self.R)

        df.fem.apply_lifting(b, [self.dR], bcs=[self.bcs], x0=[x], scale=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        df.fem.set_bc(b, self.bcs, x, -1.0)

    def solve(self):
        if self.solver is None:
            self.a = self.dR
            self.L = self.R
            self.solver = df.nls.petsc.NewtonSolver(MPI.COMM_WORLD, self)
            self.solver.atol = 1.e-8

        return self.solver.solve(self.u)


def plane_at(x):
    """ """

    def boundary(c):
        return np.isclose(c[0, :], x)

    return boundary


def gdm_error(n_elements):
    """
    ... evaluated in 2D
    """
    e = PeerlingsAnalytic()

    mesh = df.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[0, 0], [e.L / 2.0, 1]],
        [n_elements, 1],
        df.mesh.CellType.quadrilateral,
    )
    mat = GDMPlaneStrain(E=e.E, nu=0.0, ft=e.E * e.kappa0, l=e.l, dmg=damage_perfect)

    def area(coords):
        A = np.full(len(coords), 10.0)
        A[coords[:, 0] < e.W / 2.0] *= 1.0 - e.alpha
        return A

    problem = GDMProblemX(mesh, mat, fd_np=area)
    V = problem.Vd # space of displacements to apply the BCs

    b_facets_l = df.mesh.locate_entities_boundary(mesh, 1, plane_at(0.0))
    b_facets_r = df.mesh.locate_entities_boundary(mesh, 1, plane_at(e.L / 2.0))

    b_dofs_lx = df.fem.locate_dofs_topological(V.sub(0), 1, b_facets_l)
    b_dofs_ly = df.fem.locate_dofs_topological(V.sub(1), 1, b_facets_l)
    b_dofs_rx = df.fem.locate_dofs_topological(V.sub(0), 1, b_facets_r)


    u_bc = df.fem.Constant(mesh, 0.0)
    problem.bcs = [
        df.fem.dirichletbc(PETSc.ScalarType(0), b_dofs_lx, V.sub(0)),
        df.fem.dirichletbc(PETSc.ScalarType(0), b_dofs_ly, V.sub(1)),
        df.fem.dirichletbc(u_bc, b_dofs_rx, V.sub(0)),
    ]

    for t in np.linspace(0.0, 1.0, 11):
        u_bc.value = t * e.deltaL / 2.0
        iterations, converged = problem.solve()
        assert converged
        problem.update()

    # Honestly, I have not idea (yet) how to properly/easily compute the
    # errornorm to the analytic solution. Thus, we extract the dof
    # coordinates x to compute analytic dof values.
    e_fem = problem.u.sub(1).collapse()
    xs = e_fem.function_space.tabulate_dof_coordinates()[:, 0]
    e_exact = [e.e(x) for x in xs]
    return np.linalg.norm(e_fem.x.array - e_exact)


def convergence_test():
    ns = [50, 100, 200, 400]
    errors = []
    for n in ns:
        errors.append(gdm_error(n))

    ps = []
    for i in range(len(ns) - 1):
        p = np.log(errors[i] - errors[i + 1]) / np.log(1.0 / ns[i] - 1.0 / ns[i + 1])
        ps.append(p)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 3))
    plt.loglog(ns, errors, "-ko")
    plt.xlabel("# elements")
    plt.ylabel("error")
    plt.tight_layout()
    plt.savefig("gdm_convergencex.png")
    plt.show()

if __name__ == "__main__":
    convergence_test()
