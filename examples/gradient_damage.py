"""
Implicit gradient-enhanced damage model
=======================================


We solve

.. math::

    \nabla \cdot \left( (1-\omega(\kappa(\bar \varepsilon))) \boldsymbol C : \boldsymbol \varepsilon \right)  &= \boldsymbol 0  \\
    \bar \varepsilon - l^2 \Delta \bar \varepsilon &= \| \boldsymbol \varepsilon \|




"""
from helper import *
from dolfin.cpp.log import log
import matplotlib.pyplot as plt

try:
    from fenics_helpers.boundary import *
    from fenics_helpers.timestepping import TimeStepper
except Exception as e:
    print("Install fenics_helpers via (e.g.)")
    print("   pip3 install git+https://github.com/BAMResearch/fenics_helpers")
    raise (e)

"""
Building blocks
---------------

The implementation of Hookes law and the strain norm are closely tied to the
definition of the strains - i.e. 1D / 2D / 3D and how the missing dimensions
are treated (plane stress/strain). Thus, all this code is included in a single
class and not built of indivitual components.
"""


def damage_exponential(mat, k):
    k0 = mat.ft / mat.E
    a = mat.alpha
    b = mat.beta

    w = 1.0 - k0 / k * (1.0 - a + a * np.exp(b * (k0 - k)))
    dw = k0 / k * ((1.0 / k + b) * a * np.exp(b * (k0 - k)) + (1.0 - a) / k)

    return w, dw


def damage_perfect(mat, k):
    k0 = mat.ft / mat.E
    return 1.0 - k0 / k, k0 / k ** 2


class GDMPlaneStrain:
    def __init__(self):
        self.E = 20000.0
        self.nu = 0.2
        self.l = 200 ** 0.5
        self.k = 10.0
        self.ft = 2.0
        self.alpha = 0.99
        self.beta = 100.0
        self.dmg = damage_exponential
        self.kappa = None

    def eps(self, v):
        e = sym(grad(v))
        return as_vector([e[0, 0], e[1, 1], 2 * e[0, 1]])

    def integrate(self, eps_flat, e):
        self.calc_sigma(eps_flat.reshape(-1, 3), e)
        self.calc_strain_norm(eps_flat)

    def update(self, e):
        if self.kappa is None:
            self.kappa = self.ft / self.E
        self.kappa = np.maximum(e, self.kappa)

    def hooke(self):
        E, nu = self.E, self.nu
        l = E * nu / (1 + nu) / (1 - 2 * nu)
        m = E / (2.0 * (1 + nu))
        return np.array([[2 * m + l, l, 0], [l, 2 * m + l, 0], [0, 0, m]])

    def calc_strain_norm(self, eps):
        """
        eps:
            voight strains
        """
        nu, k = self.nu, self.k

        K1 = (k - 1.0) / (2.0 * k * (1.0 - 2.0 * nu))
        K2 = 3.0 / (k * (1.0 + nu) ** 2)

        exx, eyy, exy = eps[0::3], eps[1::3], eps[2::3]
        I1 = exx + eyy
        J2 = 1.0 / 6.0 * ((exx - eyy) ** 2 + exx ** 2 + eyy ** 2) + (0.5 * exy) ** 2

        A = np.sqrt(K1 ** 2 * I1 ** 2 + K2 * J2) + 1.0e-14
        self.eeq = K1 * I1 + A

        dJ2dexx = 1.0 / 3.0 * (2 * exx - eyy)
        dJ2deyy = 1.0 / 3.0 * (2 * eyy - exx)
        dJ2dexy = 0.5 * exy

        self.deeq = np.empty_like(eps)
        self.deeq[0::3] = K1 + 1.0 / (2 * A) * (2 * K1 * K1 * I1 + K2 * dJ2dexx)
        self.deeq[1::3] = K1 + 1.0 / (2 * A) * (2 * K1 * K1 * I1 + K2 * dJ2deyy)
        self.deeq[2::3] = 1.0 / (2 * A) * (K2 * dJ2dexy)

    def calc_sigma(self, eps, e):
        D = self.hooke()

        if self.kappa is None:
            self.kappa = self.ft / self.E
        k = np.maximum(e, self.kappa)
        w, dw = self.dmg(self, k)
        w_factor = 1.0 - w

        self.sigma = eps @ D * w_factor[:, None]
        self.dsigma_deps = np.tile(D.flatten(), (len(e), 1)) * w_factor[:, None]

        dk_de = (e >= k).astype(int)
        self.dsigma_de = -eps @ D * dw[:, None] * dk_de[:, None]


"""
Degrees of freedom:
    u = [d, e] 

    u .. total mixed vector
    d .. displacement field
    e .. nonlocal equivalent strain field


Momentum balance + Screened Poisson:
    Rd = eps(dd) : sigma(eps(d), e) * dx
    Re = de * e * dx + grad(de) . l ** 2 * grad(e) * dx  - de * eeq(eps) * dx

Derivatives:
    dRd/dd = eps(dd) : (dSigma_deps) * eps(d)) * dx
    dRd/de = de * (dSigma_de * eps(d)) *dx
    dRe/dd = eps(dd) * (-deeq_deps) * e * dx
    dRe/de = de * e * dx + grad(de) . l**2 * grad(e) * dx

The _trivial_ terms in the equations above are implemented using fenics
forms. The non-trivial ones are defined as functions with prefix "q_" in 
appropriately-sized quadrature function spaces and are evaluated "manually".

"""

################################################################################
#                            solution fields
################################################################################


def build_nullspace2D(V, u):
    """Function to build null space for 2D elasticity"""

    # Create list of vectors for null space
    nullspace_basis = [u.copy() for i in range(3)]
    # Build translational null space basis
    V.sub(0).dofmap().set(nullspace_basis[0], 1.0)
    V.sub(1).dofmap().set(nullspace_basis[1], 1.0)

    # Build rotational null space basis
    V.sub(0).set_x(nullspace_basis[2], -1.0, 1)
    V.sub(1).set_x(nullspace_basis[2], 1.0, 0)

    for x in nullspace_basis:
        x.apply("insert")

    # Create vector space basis and orthogonalize
    basis = VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()

    return basis


class GDM(NonlinearProblem):
    def __init__(self, mesh, mat, **kwargs):
        NonlinearProblem.__init__(self)
        self.mat = mat
        deg_d = 2
        deg_e = 2
        deg_q = 2

        metadata = {"quadrature_degree": deg_q, "quadrature_scheme": "default"}
        dxm = dx(metadata=metadata)

        cell = mesh.ufl_cell()

        # solution field
        Ed = VectorElement("CG", cell, degree=deg_d)
        Ee = FiniteElement("CG", cell, degree=deg_e)

        self.Vu = FunctionSpace(mesh, Ed * Ee)
        self.Vd, self.Ve = self.Vu.split()
        self.u = Function(self.Vu, name="d-e mixed space")

        # generic quadrature function spaces
        q = "Quadrature"
        voigt = 3
        QF = FiniteElement(q, cell, deg_q, quad_scheme="default")
        QV = VectorElement(q, cell, deg_q, quad_scheme="default", dim=voigt)
        QT = TensorElement(q, cell, deg_q, quad_scheme="default", shape=(voigt, voigt))
        VQF, VQV, VQT = [FunctionSpace(mesh, Q) for Q in [QF, QV, QT]]

        # quadrature function
        self.q_sigma = Function(VQV, name="current stresses")
        self.q_eps = Function(VQV, name="current strains")
        self.q_e = Function(VQF, name="current nonlocal equivalent strains")
        self.q_k = Function(VQF, name="current history variable kappa")
        self.q_eeq = Function(VQF, name="current (local) equivalent strain (norm)")

        self.q_dsigma_deps = Function(VQT, name="stress-strain tangent")
        self.q_dsigma_de = Function(VQV, name="stress-nonlocal-strain tangent")
        self.q_deeq_deps = Function(VQV, name="equivalent-strain-strain tangent")

        dd, de = TrialFunctions(self.Vu)
        u_, e_ = TestFunctions(self.Vu)
        d, e = split(self.u)

        try:
            f_d = kwargs["f_d"]
        except:
            f_d = Constant(1.0)

        eps = self.mat.eps
        self.R = f_d * inner(eps(u_), self.q_sigma) * dxm
        self.R += e_ * (e - self.q_eeq) * dxm
        self.R += dot(grad(e_), mat.l ** 2 * grad(e)) * dxm

        self.dR = f_d * inner(eps(dd), self.q_dsigma_deps * eps(u_)) * dxm
        self.dR += f_d * de * dot(self.q_dsigma_de, eps(u_)) * dxm
        self.dR += inner(eps(dd), -self.q_deeq_deps * e_) * dxm
        self.dR += de * e_ * dxm + dot(grad(de), mat.l ** 2 * grad(e_)) * dxm

        self.calculate_eps = LocalProjector(eps(d), VQV, dxm)
        self.calculate_e = LocalProjector(e, VQF, dxm)

        self.assembler = None
        self.null_space = build_nullspace2D(self.Vd, self.u.vector())

    def evaluate_material(self):
        # project the strain and the nonlocal equivalent strains onto
        # their quadrature spaces and ...
        self.calculate_eps(self.q_eps)
        self.calculate_e(self.q_e)

        eps_flat = self.q_eps.vector().get_local()
        e = self.q_e.vector().get_local()

        # ... "manually" evaluate_material the material.
        self.mat.integrate(eps_flat, e)

        set_q(self.q_eeq, self.mat.eeq)
        set_q(self.q_deeq_deps, self.mat.deeq)

        # Finally write the "manually" calculated values into the quadrature
        # spaces.

        set_q(self.q_sigma, self.mat.sigma)
        set_q(self.q_dsigma_deps, self.mat.dsigma_deps)
        set_q(self.q_dsigma_de, self.mat.dsigma_de)

    def update(self):
        self.calculate_e(self.q_e)
        self.mat.update(self.q_e.vector().get_local())
        set_q(self.q_k, self.mat.kappa)

    def set_bcs(self, bcs):
        # Only now can we initialize the assembler
        self.assembler = SystemAssembler(self.dR, self.R, bcs)

    def F(self, b, x):

        if not self.assembler:
            raise RuntimeError("You need to `.set_bcs(bcs)` before the solve!")
        self.evaluate_material()
        self.assembler.assemble(b, x)

    def J(self, A, x):
        self.assembler.assemble(A)
        as_backend_type(A).set_near_nullspace(self.null_space)


def bending():
    LX = 2000
    LY = 300
    LX_load = 100

    mesh = RectangleMesh(Point(0, 0), Point(LX, LY), 100, 15)
    mat = GDMPlaneStrain()
    gdm = GDM(mesh, mat)

    bcs = []
    left = point_at((0.0, 0.0), eps=0.1)
    right = point_at((LX, 0.0), eps=0.1)
    top = within_range([(LX - LX_load) / 2.0, LY], [(LX + LX_load) / 2, LY], eps=0.1)

    bc_expr = Expression("d*t", degree=0, t=0, d=-3)
    bcs.append(DirichletBC(gdm.Vd.sub(1), bc_expr, top))
    bcs.append(DirichletBC(gdm.Vd.sub(0), 0.0, left, method="pointwise"))
    bcs.append(DirichletBC(gdm.Vd.sub(1), 0.0, left, method="pointwise"))
    bcs.append(DirichletBC(gdm.Vd.sub(1), 0.0, right, method="pointwise"))

    gdm.set_bcs(bcs)

    pc = PETScPreconditioner("petsc_amg")

    # Use Chebyshev smoothing for multigrid
    PETScOptions.set("mg_levels_ksp_type", "chebyshev")
    PETScOptions.set("mg_levels_pc_type", "jacobi")

    # Improve estimate of eigenvalues for Chebyshev smoothing
    PETScOptions.set("mg_levels_esteig_ksp_type", "gmres")
    PETScOptions.set("mg_levels_ksp_chebyshev_esteig_steps", 20)

    lin_solver = PETScKrylovSolver("bicgstab", pc)

    lin_solver.parameters["nonzero_initial_guess"] = True
    lin_solver.parameters["maximum_iterations"] = 1000
    lin_solver.parameters["relative_tolerance"] = 1.0e-6
    lin_solver.parameters["error_on_nonconvergence"] = False

    lin_solver = LUSolver("mumps")
    solver = NewtonSolver(MPI.comm_world, lin_solver, PETScFactory.instance())

    # solver = NewtonSolver()
    solver.parameters["linear_solver"] = "mumps"
    solver.parameters["maximum_iterations"] = 10
    solver.parameters["error_on_nonconvergence"] = False

    def solve(t, dt):
        bc_expr.t = t
        return solver.solve(gdm, gdm.u.vector())

    ld = LoadDisplacementCurve(bcs[0])
    ld.show()
    if not ld.is_root:
        set_log_level(LogLevel.ERROR)

    fff = XDMFFile("output.xdmf")
    fff.parameters["functions_share_mesh"] = True
    fff.parameters["flush_output"] = True

    def pp(t):
        gdm.update()
        import locale

        locale.setlocale(
            locale.LC_NUMERIC, "en_US.UTF-8"
        )  # this fixes XDMF time stamps
        fff.write(gdm.u.split()[0], t)
        fff.write(gdm.u.split()[1], t)
        ld(t, assemble(gdm.R))

    TimeStepper(solve, pp, gdm.u).adaptive(1.0, dt=0.1)


class PeerlingsAnalytic(UserExpression):
    def __init__(self, **kwargs):
        self.L, self.W, self.deltaL, self.alpha = 100.0, 10.0, 0.05, 0.1
        self.E, self.kappa0, self.l = 20000.0, 1.0e-4, 1.0

        self._calculate_coeffs()
        super().__init__(**kwargs)

    def _calculate_coeffs(self):
        """
        The analytic solution is following Peerlings paper (1996) but with
        b(paper) = b^2 (here)
        g(paper) = g^2 (here)
        c(paper) = l^2 (here)
        This modification eliminates all the sqrts in the formulations.
        Plus: the formulation of the GDM in terms of l ( = sqrt(c) ) is 
        more common in modern pulbications.
        """

        # imports only used here...
        from sympy import Symbol, symbols, N, integrate, cos, exp, lambdify
        import scipy.optimize

        # unknowns
        x = Symbol("x")
        unknowns = symbols("A1, A2, B1, B2, C, b, g, w")
        A1, A2, B1, B2, C, b, g, w = unknowns

        l = self.l
        kappa0 = self.kappa0

        # 0 <= x <= W/2
        e1 = C * cos(g / l * x)
        # W/2 <  x <= w/2
        e2 = B1 * exp(b / l * x) + B2 * exp(-b / l * x)
        # w/2 <  x <= L/2
        e3 = A1 * exp(x / l) + A2 * exp(-x / l) + (1 - b * b) * kappa0

        de1, de2, de3 = e1.diff(x), e2.diff(x), e3.diff(x)

        eq1 = N(e1.subs(x, self.W / 2) - e2.subs(x, self.W / 2))
        eq2 = N(de1.subs(x, self.W / 2) - de2.subs(x, self.W / 2))
        eq3 = N(e2.subs(x, w / 2) - kappa0)
        eq4 = N(de2.subs(x, w / 2) - de3.subs(x, w / 2))
        eq5 = N(e3.subs(x, w / 2) - kappa0)
        eq6 = N(de3.subs(x, self.L / 2))
        eq7 = N((1 - self.alpha) * (1 + g * g) - (1 - b * b))
        eq8 = N(
            integrate(e1, (x, 0, self.W / 2))
            + integrate(e2, (x, self.W / 2, w / 2))
            + integrate(e3, (x, w / 2, self.L / 2))
            - self.deltaL / 2
        )

        eqs = [
            lambdify(unknowns, eq) for eq in [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8]
        ]

        def global_func(x):
            return np.array([eqs[i](*x) for i in range(8)])

        result = scipy.optimize.root(
            global_func, [0.0, 5e2, 3e-7, 7e-3, 3e-3, 3e-1, 2e-1, 4e1]
        )
        if not result["success"]:
            raise RuntimeError(
                "Could not find the correct coefficients. Try to tweak the initial values."
            )

        self.coeffs = result["x"]

    def e(self, x):
        A1, A2, B1, B2, C, b, g, w = self.coeffs
        if x <= self.W / 2.0:
            return C * np.cos(g / self.l * x)
        elif x <= w / 2.0:
            return B1 * np.exp(b / self.l * x) + B2 * np.exp(-b / self.l * x)
        else:
            return (
                (1.0 - b * b) * self.kappa0
                + A1 * np.exp(x / self.l)
                + A2 * np.exp(-x / self.l)
            )

    def eval(self, value, x):
        value[0] = self.e(x[0])


def gdm_error(n_elements):
    """
    ... evaluated in 2D
    """
    sol = PeerlingsAnalytic(degree=4)

    mesh = RectangleMesh(Point(0.0, 0.0), Point(sol.L / 2.0, 1), n_elements, 1)
    mat = GDMPlaneStrain()
    mat.E = sol.E
    mat.nu = 0.0
    mat.ft = sol.E * sol.kappa0
    mat.l = sol.l
    mat.dmg = damage_perfect

    area = Expression(
        "x[0] <= W/2. ? 10.0 * (1. - a) : 10.0", W=sol.W, a=sol.alpha, degree=0
    )
    gdm = GDM(mesh, mat, f_d=area)

    bc_expr = Expression("t*d", degree=0, t=0, d=sol.deltaL / 2)
    bc0 = DirichletBC(gdm.Vd, (0.0, 0.0), plane_at(0.0))
    bc1 = DirichletBC(gdm.Vd.sub(0), bc_expr, plane_at(sol.L / 2.0))
    gdm.set_bcs([bc0, bc1])

    solver = NewtonSolver()
    solver.parameters["linear_solver"] = "mumps"
    solver.parameters["maximum_iterations"] = 10
    solver.parameters["error_on_nonconvergence"] = False

    for t in np.linspace(0.0, 1.0, 11):
        print("Solving for t =", t)
        bc_expr.t = t
        assert solver.solve(gdm, gdm.u.vector())[1]

    return errornorm(sol, gdm.u.split()[1])


def peerlings1d():
    ns = [50, 100, 200, 400]
    errors = []
    for n in ns:
        errors.append(gdm_error(n))

    ps = []
    for i in range(len(ns) - 1):
        p = np.log(errors[i] - errors[i + 1]) / np.log(1.0 / ns[i] - 1.0 / ns[i + 1])
        ps.append(p)

    plt.loglog(ns, errors)
    plt.show()

    print(ps)


if __name__ == "__main__":
    # print(gdm_error(200))
    assert gdm_error(200) < 1.0e-8
    bending()
    # peerlings1d()
    list_timings(TimingClear.keep, [TimingType.wall])
