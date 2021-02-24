import dolfin as df
import numpy as np
from fenics_helpers import boundary
from fenics_helpers.timestepping import TimeStepper
import constitutive as c

class GDMProblem(c.MechanicsProblem):
    def __init__(self, mesh, prm, law):
        df.NonlinearProblem.__init__(self)

        self.mesh = mesh
        self.prm = prm

        self.law = law

        self.base = c.BaseGDM(self.law)

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
        self.base.resize(n_gauss_points);

        dd, de = df.TrialFunctions(self.V)
        d_, e_ = df.TestFunctions(self.V)
        d, e = df.split(self._u)
        self.d = d


        n_gauss_points = len(self.q_eps.vector().get_local()) // c.q_dim(prm.constraint)
        self.base.resize(n_gauss_points);

        eps = self.eps
        f_d = 1.
        self.R = f_d * df.inner(eps(d_), self.q_sigma) * self.dxm
        self.R += e_ * (e - self.q_eeq) * self.dxm
        self.R += df.dot(df.grad(e_), prm.l ** 2 * df.grad(e)) * self.dxm

        self.dR = f_d * df.inner(eps(dd), self.q_dsigma_deps * eps(d_)) * self.dxm
        self.dR += f_d * de * df.dot(self.q_dsigma_de, eps(d_)) * self.dxm
        self.dR += df.inner(eps(dd), -self.q_deeq_deps * e_) * self.dxm
        self.dR += de * e_ * self.dxm + df.dot(df.grad(de), prm.l ** 2 * df.grad(e_)) * self.dxm

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

    def evaluate_material(self):
        # project the strain and the nonlocal equivalent strains onto
        # their quadrature spaces and ...
        self.calculate_eps(self.q_eps)
        self.calculate_e(self.q_e)
        self.base.evaluate(self.q_eps.vector().get_local(), self.q_e.vector().get_local())

        # ... and write the calculated values into their quadrature spaces.
        c.helper.set_q(self.q_sigma, self.law.get("sigma"))
        c.helper.set_q(self.q_dsigma_deps, self.law.get("dsigma_deps"))
        c.helper.set_q(self.q_deeq_deps, self.law.get("deeq"))
        c.helper.set_q(self.q_dsigma_de, self.law.get("dsigma_de"))
        c.helper.set_q(self.q_eeq, self.law.get("eeq"))

    def update(self):
        self.calculate_eps(self.q_eps)
        self.calculate_e(self.q_e)
        self.base.update(self.q_eps.vector().get_local(), self.q_e.vector().get_local())


if __name__ == "__main__":
    # return
    LX = 2000
    LY = 300
    LX_load = 100

    prm = c.Parameters(c.Constraint.PLANE_STRAIN)
    prm.E = 20000.
    prm.nu = 0.2
    prm.l = 200**0.5
    prm.ft = 2.
    prm.k = 10.0
    prm.alpha = 0.99
    prm.beta = 100.0

    prm.deg_d = 2
    law = c.GradientDamage(prm.E, prm.nu, prm.constraint, prm.ft, prm.alpha, prm.beta, prm.k)

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
