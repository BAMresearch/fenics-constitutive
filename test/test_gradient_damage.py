import dolfin as df
import numpy as np
from fenics_helpers import boundary
from fenics_helpers.timestepping import TimeStepper
import constitutive as c


class GDMSpaces:
    def __init__(self, constraint):
        self.deg_d = 2
        self.deg_e = 2
        self.deg_q = 2
        self.constraint = constraint

    def create(self, mesh, mesh_function=None):
        self.mesh = mesh
        self.mesh_function = mesh_function

        self.metadata = {
            "quadrature_degree": self.deg_q,
            "quadrature_scheme": "default",
        }
        self.dxm = df.dx(metadata=self.metadata, subdomain_data=mesh_function)

        # solution field
        Ed = df.VectorElement("CG", mesh.ufl_cell(), degree=self.deg_d)
        Ee = df.FiniteElement("CG", mesh.ufl_cell(), degree=self.deg_d)

        self.V = df.FunctionSpace(mesh, Ed * Ee)
        self.Vd, self.Ve = self.V.split()

        self.dd, self.de = df.TrialFunctions(self.V)
        self.d_, self.e_ = df.TestFunctions(self.V)

        self.u = df.Function(self.V, name="d-e mixed space")
        self.d, self.e = df.split(self.u)

        # generic quadrature function spaces
        VQF, VQV, VQT = c.helper.spaces(mesh, self.deg_q, c.q_dim(self.constraint))

        # quadrature function
        self.q_sigma = df.Function(VQV, name="current stresses")
        self.q_eps = df.Function(VQV, name="current strains")
        self.q_e = df.Function(VQF, name="current nonlocal equivalent strains")
        self.q_k = df.Function(VQF, name="current history variable kappa")
        self.q_eeq = df.Function(VQF, name="current (local) equivalent strain (norm)")

        self.q_dsigma_deps = df.Function(VQT, name="stress-strain tangent")
        self.q_dsigma_de = df.Function(VQV, name="stress-nonlocal-strain tangent")
        self.q_deeq_deps = df.Function(VQV, name="equivalent-strain-strain tangent")

        self.n = len(self.q_eps.vector().get_local()) // c.q_dim(self.constraint)
        self.nq = self.n // mesh.num_cells()

        self.ip_flags = np.repeat(mesh_function.array(), self.nq)
        

    def eps(self, u):
        e = df.sym(df.grad(u))
        dim = self.mesh.geometric_dimension()
        if dim == 1:
            return df.as_vector([e[0, 0]])

        if dim == 2:
            return df.as_vector([e[0, 0], e[1, 1], 2 * e[0, 1]])

        if dim == 3:
            return df.as_vector(
                [e[0, 0], e[1, 1], e[2, 2], 2 * e[1, 2], 2 * e[0, 2], 2 * e[0, 1]]
            )


# def gdm_form


class GDMProblem(c.MechanicsProblem):
    def __init__(self, mesh, prm, loop):
        df.NonlinearProblem.__init__(self)

        self.mesh = mesh
        self.prm = prm

        if mesh.geometric_dimension() != c.g_dim(prm.constraint):
            raise RuntimeError(
                f"The geometric dimension of the mesh does not match the {prm.constraint} constraint."
            )

        metadata = {"quadrature_degree": prm.deg_q, "quadrature_scheme": "default"}
        self.dxm = df.dx(metadata=metadata)

        # solution field
        Ed = df.VectorElement("CG", mesh.ufl_cell(), degree=prm.deg_d)
        Ee = df.FiniteElement("CG", mesh.ufl_cell(), degree=prm.deg_d)

        self.V = df.FunctionSpace(mesh, Ed * Ee)
        self._u = df.Function(self.V, name="d-e mixed space")

        # generic quadrature function spaces
        VQF, VQV, VQT = c.helper.spaces(mesh, prm.deg_q, c.q_dim(prm.constraint))

        # quadrature function
        self.q_sigma = df.Function(VQV, name="current stresses")
        self.q_eps = df.Function(VQV, name="current strains")
        self.q_e = df.Function(VQF, name="current nonlocal equivalent strains")
        self.q_k = df.Function(VQF, name="current history variable kappa")
        self.q_eeq = df.Function(VQF, name="current (local) equivalent strain (norm)")

        self.q_dsigma_deps = df.Function(VQT, name="stress-strain tangent")
        self.q_dsigma_de = df.Function(VQV, name="stress-nonlocal-strain tangent")
        self.q_deeq_deps = df.Function(VQV, name="equivalent-strain-strain tangent")

        n_gauss_points = len(self.q_eps.vector().get_local()) // c.q_dim(prm.constraint)

        self.loop = loop
        self.loop.resize(n_gauss_points)

        dd, de = df.TrialFunctions(self.V)
        d_, e_ = df.TestFunctions(self.V)
        d, e = df.split(self._u)
        self.d = d

        eps = self.eps
        f_d = 1.0
        self.R = f_d * df.inner(eps(d_), self.q_sigma) * self.dxm
        self.R += e_ * (e - self.q_eeq) * self.dxm
        self.R += df.dot(df.grad(e_), prm.l ** 2 * df.grad(e)) * self.dxm

        self.dR = f_d * df.inner(eps(dd), self.q_dsigma_deps * eps(d_)) * self.dxm
        self.dR += f_d * de * df.dot(self.q_dsigma_de, eps(d_)) * self.dxm
        self.dR += df.inner(eps(dd), -self.q_deeq_deps * e_) * self.dxm
        self.dR += (
            de * e_ * self.dxm
            + df.dot(df.grad(de), prm.l ** 2 * df.grad(e_)) * self.dxm
        )

        self.calculate_eps = c.helper.LocalProjector(eps(self.d), VQV, self.dxm)
        self.calculate_e = c.helper.LocalProjector(e, VQF, self.dxm)

        self._assembler = None
        self._bcs = None

    @property
    def u(self):
        return self._u

    @property
    def Vd(self):
        return self.V.split()[0]

    @property
    def Ve(self):
        return self.V.split()[1]

    # @profile
    def evaluate_material(self):
        # project the strain and the nonlocal equivalent strains onto
        # their quadrature spaces and ...
        self.calculate_eps(self.q_eps)
        self.calculate_e(self.q_e)
        self.loop.evaluate(
            self.q_eps.vector().get_local(), self.q_e.vector().get_local()
        )

        # ... and write the calculated values into their quadrature spaces.
        c.helper.set_q(self.q_sigma, self.loop.get(c.Q.SIGMA))
        c.helper.set_q(self.q_dsigma_deps, self.loop.get(c.Q.DSIGMA_DEPS))
        c.helper.set_q(self.q_deeq_deps, self.loop.get(c.Q.DEEQ))
        c.helper.set_q(self.q_dsigma_de, self.loop.get(c.Q.DSIGMA_DE))
        c.helper.set_q(self.q_eeq, self.loop.get(c.Q.EEQ))

    def update(self):
        self.calculate_eps(self.q_eps)
        self.calculate_e(self.q_e)
        self.loop.update(self.q_eps.vector().get_local(), self.q_e.vector().get_local())


def tensile_meso():
    import matplotlib.pyplot as plt

    mesh = df.Mesh()
    mvc = df.MeshValueCollection("size_t", mesh, 1)
    LX, LY = 80.0, 80.0  # magic!

    with df.XDMFFile("mesh.xdmf") as f:
        f.read(mesh)
        f.read(mvc, "gmsh:physical")

    subdomains = df.MeshFunction("size_t", mesh, mvc)

    # df.plot(subdomains)
    # plt.show()

    mat_l = 2.0

    s = GDMSpaces(c.Constraint.PLANE_STRAIN)
    s.create(mesh, subdomains)
    R = df.inner(s.eps(s.d_), s.q_sigma) * s.dxm(1)
    dR = df.inner(s.eps(s.dd), s.q_dsigma_deps * s.eps(s.d_)) * s.dxm(1)

    R += s.e_ * (s.e - s.q_eeq) * s.dxm(1)
    R += df.dot(df.grad(s.e_), mat_l ** 2 * df.grad(s.e)) * s.dxm(1)

    dR += s.de * df.dot(s.q_dsigma_de, s.eps(s.d_)) * s.dxm(1)
    dR += df.inner(s.eps(s.dd), -s.q_deeq_deps * s.e_) * s.dxm(1)
    dR += s.de * s.e_ * s.dxm(1)
    dR += df.dot(df.grad(s.de), mat_l ** 2 * df.grad(s.e_)) * s.dxm(1)

    R += df.inner(s.eps(s.d_), s.q_sigma) * s.dxm(2)
    dR += df.inner(s.eps(s.dd), s.q_dsigma_deps * s.eps(s.d_)) * s.dxm(2)
    dR += s.de * s.e_ * s.dxm(2)

    R += df.inner(s.eps(s.d_), s.q_sigma) * s.dxm(3)
    dR += df.inner(s.eps(s.dd), s.q_dsigma_deps * s.eps(s.d_)) * s.dxm(3)
    dR += s.de * s.e_ * s.dxm(3)
        
    VQF, VQV, VQT = c.helper.spaces(s.mesh, s.deg_q, c.q_dim(s.constraint))
    calculate_eps = c.helper.LocalProjector(s.eps(s.d), VQV, s.dxm)
    calculate_e = c.helper.LocalProjector(s.e, VQF, s.dxm(1))


    F = 0.5 # interface reduction
    t = 0.5 # interface thickness
    lawAggreg = c.LinearElasticNew(2*26738, 0.18, s.constraint)
    lawInterf= c.LocalDamageNew(F*26738, 0.18, s.constraint, F * 3.4, 0.99, F/t*0.12, 10.)
    lawMatrix= c.GradientDamage(26738, 0.18, s.constraint, 3.4, 0.99, 0.0216, 10.)

    loop = c.IpLoop()
    loop.add_law(lawMatrix, np.where(s.ip_flags == 1)[0])
    loop.add_law(lawAggreg, np.where(s.ip_flags == 2)[0])
    loop.add_law(lawInterf, np.where(s.ip_flags == 3)[0])
    loop.resize(s.n)
   
    bot = boundary.plane_at(0, "y")
    top = boundary.plane_at(LY, "y")
    bc_expr = df.Expression("u", degree=0, u=0)
    bcs = []
    bcs.append(df.DirichletBC(s.Vd.sub(1), bc_expr, top))
    bcs.append(df.DirichletBC(s.Vd.sub(1), 0.0, bot))
    bcs.append(df.DirichletBC(s.Vd.sub(0), 0.0, boundary.point_at((0,0)), method="pointwise"))

    
    # return

    assembler = df.SystemAssembler(dR, R, bcs)

    class SolveMe(df.NonlinearProblem):
        def F(self, b, x):
            calculate_eps(s.q_eps)
            calculate_e(s.q_e)
            loop.evaluate(s.q_eps.vector().get_local(), s.q_e.vector().get_local())

            # ... and write the calculated values into their quadrature spaces.
            c.helper.set_q(s.q_sigma, loop.get(c.Q.SIGMA))
            c.helper.set_q(s.q_dsigma_deps, loop.get(c.Q.DSIGMA_DEPS))
            c.helper.set_q(s.q_deeq_deps, loop.get(c.Q.DEEQ))
            c.helper.set_q(s.q_dsigma_de, loop.get(c.Q.DSIGMA_DE))
            c.helper.set_q(s.q_eeq, loop.get(c.Q.EEQ))

            assembler.assemble(b, x)

        def J(self, A, x):
            assembler.assemble(A)

    linear_solver = df.LUSolver("mumps")
    solver = df.NewtonSolver(
        df.MPI.comm_world, linear_solver, df.PETScFactory.instance()
    )
    solver.parameters["linear_solver"] = "mumps"
    solver.parameters["maximum_iterations"] = 10
    solver.parameters["error_on_nonconvergence"] = False

    problem=SolveMe()

    def solve(t, dt):
        print(t, dt)
        bc_expr.u = 0.1 * t
        # try:
        return solver.solve(problem, s.u.vector())
        # except:
        # return -1, False

    ld = c.helper.LoadDisplacementCurve(bcs[0])
    ld.show()
    if not ld.is_root:
        df.set_log_level(df.LogLevel.ERROR)

    fff = df.XDMFFile("output.xdmf")
    fff.parameters["functions_share_mesh"] = True
    fff.parameters["flush_output"] = True

    plot_space = df.FunctionSpace(s.mesh, "DG", 0)
    k = df.Function(plot_space, name="kappa")

    def pp(t):
        calculate_eps(s.q_eps)
        calculate_e(s.q_e)
        loop.update(s.q_eps.vector().get_local(), s.q_e.vector().get_local())

        # this fixes XDMF time stamps
        import locale

        locale.setlocale(locale.LC_NUMERIC, "en_US.UTF-8")
        d, e = s.u.split(0)
        d.rename("disp", "disp")
        e.rename("e", "e")

        all_kappa = lawInterf.kappa() + lawMatrix.kappa()
        k.vector().set_local(all_kappa[::s.nq])

        fff.write(d, t)
        fff.write(e, t)
        fff.write(k, t)

        ld(t, df.assemble(R))

    TimeStepper(solve, pp, s.u).adaptive(1.0, dt=0.1)

    pass


def bending():
    # return
    LX = 2000
    LY = 300
    LX_load = 100

    prm = c.Parameters(c.Constraint.PLANE_STRAIN)
    prm.E = 20000.0
    prm.nu = 0.2
    prm.l = 200 ** 0.5
    prm.ft = 2.0
    prm.k = 10.0
    prm.alpha = 0.99
    prm.beta = 100.0

    prm.deg_d = 2
    law = c.GradientDamage(
        prm.E, prm.nu, prm.constraint, prm.ft, prm.alpha, prm.beta, prm.k
    )

    mesh = df.RectangleMesh(df.Point(0, 0), df.Point(LX, LY), 100, 15)
    problem = GDMProblem(mesh, prm, law)

    left = boundary.point_at((0.0, 0.0), eps=0.1)
    right = boundary.point_at((LX, 0.0), eps=0.1)
    top = boundary.within_range(
        [(LX - LX_load) / 2.0, LY], [(LX + LX_load) / 2, LY], eps=0.1
    )
    bc_expr = df.Expression("d*t", degree=0, t=0, d=-3)
    bcs = []
    bcs.append(df.DirichletBC(problem.Vd.sub(1), bc_expr, top))
    bcs.append(df.DirichletBC(problem.Vd.sub(0), 0.0, left, method="pointwise"))
    bcs.append(df.DirichletBC(problem.Vd.sub(1), 0.0, left, method="pointwise"))
    bcs.append(df.DirichletBC(problem.Vd.sub(1), 0.0, right, method="pointwise"))

    # everywhere = boundary.point_at((0,0), eps=1e6)
    # bcs.append(df.DirichletBC(problem.Ve, 0.0, everywhere, method="pointwise"))

    problem.set_bcs(bcs)

    linear_solver = df.LUSolver("mumps")
    solver = df.NewtonSolver(
        df.MPI.comm_world, linear_solver, df.PETScFactory.instance()
    )
    solver.parameters["linear_solver"] = "mumps"
    solver.parameters["maximum_iterations"] = 10
    solver.parameters["error_on_nonconvergence"] = False

    def solve(t, dt):
        print(t, dt)
        bc_expr.t = t
        # try:
        return solver.solve(problem, problem.u.vector())
        # except:
        # return -1, False

    ld = c.helper.LoadDisplacementCurve(bcs[0])
    ld.show()
    if not ld.is_root:
        set_log_level(LogLevel.ERROR)

    fff = df.XDMFFile("output.xdmf")
    fff.parameters["functions_share_mesh"] = True
    fff.parameters["flush_output"] = True

    def pp(t):
        problem.update()

        # this fixes XDMF time stamps
        import locale

        locale.setlocale(locale.LC_NUMERIC, "en_US.UTF-8")
        fff.write(problem.u, t)

        ld(t, df.assemble(problem.R))

    TimeStepper(solve, pp, problem.u).adaptive(1.0, dt=0.1)


if __name__ == "__main__":
    # bending()
    tensile_meso()
