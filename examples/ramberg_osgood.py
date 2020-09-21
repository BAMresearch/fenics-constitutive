"""
Ramberg Osgood material law
===========================

Introduction and governing equations
------------------------------------
The ramberg osgood material law can be used to model
ductile behaviour for monotonic loading. In contrast to
incremental plasticity models stress and strain are directly
related and thus the ramberg osgood model is in fact a nonlinear elastic model.
While algorithmically the solution of the ramberg osgood constitutive law
in a FE code is rather simple, it involves the solution of a power law (on integration
point level) which can be challenging with regard to the implementation in FEniCS.

Linearized principle of virtual power:

.. math::
    \int_\Omega \bm \varepsilon \cdot
    \frac{\partial \bm\sigma}{\partial \bm\varepsilon} \cdot \bm \varepsilon \;\mathrm{d}x
    = f_{\mathrm{ext}} - \int_\Omega \bm \sigma \cdot \bm \varepsilon \;\mathrm{d}x


Constitutive law
****************
For a complete derivation of the equations we refer to XX and only summarize the ones essential for the
presented implementation.
The strain is given by

.. math::

    \bm{\varepsilon} = \frac{1}{3K} (\bm{\sigma} \cdot \bm I) \bm{I} + \left(
    \frac{1}{2G} + \frac{3\alpha}{2E} {\left( \frac{\sigma_{\mathrm{v}}}{\sigma_{\mathrm{y}}} \right)}^{n-1}
    \right) \bm{\sigma'},

where the stress deviator is denoted by $\bm \sigma'$ and the equivalent stress is

.. math::

    \sigma_{\mathrm{v}} = \sqrt{\frac{3}{2} \bm \sigma' \cdot \bm \sigma'}.

$E, \nu, \alpha, n$ and $\sigma_{\mathrm{y}}$ are material parameters (bulk modulus $K$ and
shear modulus $G$ are given in terms of $E$ and $\nu$).

Inversion of the strain stress relation:

.. math::
    \bm \sigma = \frac{2 \sigma_{\mathrm{v}}}{3 \varepsilon_{\mathrm{v}}}
    \bm \varepsilon' + \frac{K}{3} (\bm\varepsilon \cdot \bm I) \bm I

Consistent tangent:

.. math::
    \frac{\partial \bm \sigma}{\partial \bm \varepsilon} =
    \frac{2\sigma_{\mathrm{v}}}{3\varepsilon_{\mathrm{v}}}\left(
    \bm I - \frac{2}{3\varepsilon_{\mathrm{v}}}\left(
    \frac{1}{\varepsilon_{\mathrm{v}}} - \frac{1}{
    \frac{\sigma_{\mathrm{v}}}{3G} + \alpha n \frac{\sigma_{\mathrm{y}}}{E} {\left(\frac{\sigma_{\mathrm{v}}}{\sigma_{\mathrm{y}}}\right)}^{n}
    }
    \right)\bm{\varepsilon}' \circ \bm{\varepsilon}'
    \right)
    + \frac{1}{3}\left(K - \frac{2\sigma_{\mathrm{v}}}{3 \varepsilon_{\mathrm{v}}}\right) \bm{I} \circ \bm{I}


"""

import sys
from pathlib import Path

from helper import *
from dolfin.cpp.log import log

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

"""
Solution of the constitutive law
--------------------------------

The solution of the power law mentioned above makes a vectorization of the
numpy code difficult. Hence we could use a C++ function/class to solve the constitutive law.
Another option is the use of `numba <https://numba.pydata.org/>`_ to speed things up
which is done here.
"""

import numba

@numba.njit
def solve_ramberg_osgood(E, NU, ALPHA, NEXP, SIGY, NGAUSS, GDIM, STRAIN, STRESS):
    """solve ramberg osgood constitutive equation over entire
    computational domain

    Parameters
    ----------

    E, NU, ALPHA, NEXP, SIGY : material parameters
    NGAUSS : total number of gauss points
    GDIM : geometrical dimension
    STRAIN :
        strain value of each cell in Omega
    STRESS :
        previous stress state

    Returns
    -------
    STRESS : np.ndarray
        stress for each cell
    DDSDDE : np.ndarray
        tangent for each cell
    """

    LAMBDA = E * NU / (1 + NU) / (1 - 2 * NU)
    MU = E / (2 * (1 + NU))
    K = E / (1 - 2 * NU)  # bulk modulus

    DDSDDE = np.zeros((NGAUSS, GDIM * 2, GDIM * 2))

    if GDIM == 2:
        Cel = np.array(
            [
                [LAMBDA + 2 * MU, LAMBDA, LAMBDA, 0.0],
                [LAMBDA, LAMBDA + 2 * MU, LAMBDA, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2 * MU],
            ]
        )
    elif GDIM == 3:
        Cel = np.array(
            [
                [LAMBDA + 2 * MU, LAMBDA, LAMBDA, 0.0, 0.0, 0.0],
                [LAMBDA, LAMBDA + 2 * MU, LAMBDA, 0.0, 0.0, 0.0],
                [LAMBDA, LAMBDA, LAMBDA + 2 * MU, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2 * MU, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 2 * MU, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 2 * MU],
            ]
        )

    zero_strain_tolerance = 1e-12
    sv_tol = 1e-12
    maxiter = 50
    # helpers voigt notation
    I2 = np.zeros(GDIM * 2, dtype=np.double)  # Identity of rank 2 tensor
    I2[0] = 1.0
    I2[1] = 1.0
    I2[2] = 1.0
    I4 = np.eye(GDIM * 2, dtype=np.double)  # Identity of rank 4 tensor

    for n in range(NGAUSS):
        # strain at time t + delta t
        eps = STRAIN[n]
        tr_eps = np.sum(eps[:3])
        eps_dev = eps - tr_eps * I2 / 3
        #  assert np.isclose(0.0, np.sum(eps_dev[:3]))
        ev = np.sqrt(2.0 / 3.0 * np.dot(eps_dev, eps_dev))

        if ev < zero_strain_tolerance:
            # return elastic tangent
            # --> do nothing for tangent
            STRESS[n] = Cel @ eps
            DDSDDE[n] = Cel
        else:
            # compute correct tangent and stress
            # stress at time t
            sig = STRESS[n]
            tr_sig = np.sum(sig[:3])
            sig_dev = sig - tr_sig * I2 / 3
            #  assert np.isclose(0.0, np.sum(sig_dev[:3]))
            # equivalent stress at time t is used as initial guess
            sv_initial = np.sqrt(3.0 / 2.0 * np.dot(sig_dev, sig_dev))

            # stress at time t + delta t
            if sv_initial <= SIGY:
                sv = sv_initial
            else:
                # initial_guess is > sigy
                sv = (SIGY ** (NEXP - 1.0) * E * ev / ALPHA) ** (1.0 / NEXP)

            def f(x):
                stuff = 1.0 / (2.0 * MU) + 3.0 / (2.0 * E) * ALPHA * (x / SIGY) ** (
                    NEXP - 1.0
                )
                return stuff * 2.0 / 3.0 * x - ev

            def df(x):
                return 1.0 / (3.0 * MU) + NEXP * ALPHA / E * (x / SIGY) ** (NEXP - 1.0)

            s = f(sv)
            ds = df(sv)

            niter = 0
            while abs(f(sv)) > sv_tol:
                sv = sv - s / ds
                s = f(sv)
                ds = df(sv)
                niter += 1
                if niter > maxiter:
                    break

            sig_dev = 2.0 * sv / 3.0 / ev * eps_dev
            tr_sig = K * tr_eps
            sig = tr_sig * I2 / 3.0 + sig_dev
            STRESS[n] = sig

            nenner = sv / (3.0 * MU) + ALPHA * NEXP * SIGY / E * ((sv / SIGY) ** (NEXP))
            tangent = 2 * sv / 3 / ev * (
                I4
                - 2.0
                / 3.0
                / ev
                * (1.0 / ev - 1.0 / nenner)
                * np.outer(eps_dev, eps_dev)
            ) + 1.0 / 3.0 * (K - 2 * sv / (3 * ev)) * np.outer(I2, I2)
            DDSDDE[n] = tangent
    return STRESS, DDSDDE



"""
Examples
--------

Simple Tension Test
*******************

To test the above implementation we compare our numerical
results to the analytical solution for a (simple) tension test
in 2D.

"""

class AnalyticalSolution:
    """base class for ramberg osgood material solutions"""

    def __init__(self, max_load, **kwargs):
        self.load = max_load
        self.E = kwargs.get("E", 210e3)
        self.NU = kwargs.get("NU", 0.3)
        self.ALPHA = kwargs.get("ALPHA", 0.01)
        self.N = kwargs.get("N", 5.0)
        self.K = self.E / (1.0 - 2.0 * self.NU)
        self.G = self.E / 2.0 / (1.0 + self.NU)
        self.SIGY = kwargs.get("SIGY", 500.0)

    def energy(self):
        assert np.sum(self.sigma) > 0.0
        return np.trapz(self.sigma, self.eps)


class SimpleTensionSolution2D(AnalyticalSolution):
    """analytical solution for simple tension in 2D"""

    def __init__(self, max_load, **kwargs):
        super().__init__(max_load, **kwargs)

    def solve(self):
        from scipy.optimize import newton
        from sympy import symbols, Derivative, lambdify, sqrt

        E = self.E
        K = self.K
        G = self.G
        ALPHA = self.ALPHA
        SIGY = self.SIGY
        N = self.N

        def f(x, s):
            """equation to solve is eps33(x, s) = 0
            x:      sigma33
            s:      sigma22 (given as tension direction)
            """
            return (x + s) / 3.0 / K + (
                1.0 / 2.0 / G
                + 3.0
                * ALPHA
                / 2.0
                / E
                * (np.sqrt((s - x) ** 2 + x * s) / SIGY) ** (N - 1.0)
            ) * (2.0 * x - s) / 3.0

        x, s = symbols("x s")
        f_sym = (x + s) / 3.0 / K + (
            1.0 / 2.0 / G
            + 3.0 * ALPHA / 2.0 / E * (sqrt((s - x) ** 2 + x * s) / SIGY) ** (N - 1.0)
        ) * (2.0 * x - s) / 3.0
        Df = Derivative(f_sym, x)
        df = lambdify((x, s), Df.doit(), "numpy")

        s = np.linspace(0, self.load)  # sigma22
        x = np.zeros_like(s)  # initial guess
        s33 = newton(f, x, fprime=df, args=(s,), tol=1e-12)

        e11 = (s + s33) / 3.0 / K + (
            1.0 / 2.0 / G
            + 3.0
            * ALPHA
            / 2.0
            / E
            * (np.sqrt((s - s33) ** 2 + s * s33) / SIGY) ** (N - 1.0)
        ) * (-(s33 + s)) / 3.0
        e22 = (s + s33) / 3.0 / K + (
            1.0 / 2.0 / G
            + 3.0
            * ALPHA
            / 2.0
            / E
            * (np.sqrt((s - s33) ** 2 + s * s33) / SIGY) ** (N - 1.0)
        ) * (2.0 * s - s33) / 3.0
        self.sigma = s
        self.eps = e22
        return e11, e22, s

def main(args):
    args = parse_arguments(args)

    with open(root / "material.yml", "r") as infile:
        matparam = yaml.safe_load(infile)

    #  test the impl of the constitutive equations for analytical strain
    if args["--test"]:
        for gdim in (2, 3):
            if gdim == 2:
                assert test_material_shear(args, matparam, gdim)
            assert test_material_tension(args, matparam, gdim)
        print("tests are a success")

    # do simple tension test
    n = args["N"]
    mesh = df.UnitSquareMesh(n, n)

    test = simple_tension(args, mesh, matparam)
    assert test

    # timings
    if args["--timeit"]:
        results = root / "results"
        args["--write"] = False

        timings = []
        for _ in range(5):
            t0 = time()
            test = simple_tension(args, mesh, matparam)
            t = time() - t0
            timings.append(t)
        np.save(results / f"numpy_jitted_time_{args['N']}.npy", np.array(timings))


def test_material_shear(args, matparam, gdim):
    E = matparam["E"]
    NU = matparam["NU"]
    ALPHA = matparam["ALPHA"]
    NEXP = matparam["N"]
    SIGY = matparam["SIGY"]
    NGAUSS = 1
    LAMBDA = E * NU / (1 + NU) / (1 - 2 * NU)
    MU = E / (2 * (1 + NU))
    GDIM = gdim

    ana = SimpleShearSolution2D(1000.0, **matparam)
    e12, s12 = ana.solve()
    e11 = np.zeros(e12.size)
    e22 = np.zeros(e11.size)
    e33 = np.zeros(e11.size)

    eps = np.empty((1, gdim * 2), dtype=np.float)
    stress = np.zeros(eps.shape)
    bools = []
    num_sig = []
    for k in range(e22.size):
        strain = np.zeros(gdim * 2)
        strain[3] = np.sqrt(2) * e12[k]
        eps[0] = strain
        sigma, ddsdde = solve_ramberg_osgood(
            E, NU, ALPHA, NEXP, SIGY, NGAUSS, gdim, eps, stress
        )
        num_sig.append(sigma[0][3])
        bools.append(np.isclose(sigma[0][3], np.sqrt(2) * s12[k]))

    return all(bools)


def test_material_tension(args, matparam, gdim):
    E = matparam["E"]
    NU = matparam["NU"]
    ALPHA = matparam["ALPHA"]
    NEXP = matparam["N"]
    SIGY = matparam["SIGY"]
    NGAUSS = 1
    LAMBDA = E * NU / (1 + NU) / (1 - 2 * NU)
    MU = E / (2 * (1 + NU))
    GDIM = gdim

    if gdim == 3:
        ana = SimpleTensionSolution3D(2718.0, **matparam)
        e11, e22, e33, s22 = ana.solve()
    elif gdim == 2:
        ana = SimpleTensionSolution2D(2718.0, **matparam)
        e11, e22, s22 = ana.solve()
        e33 = np.zeros(e11.size)

    eps = np.empty((1, gdim * 2), dtype=np.float)
    stress = np.zeros(eps.shape)
    bools = []
    for k in range(e22.size):
        strain = np.zeros(gdim * 2)
        strain[0] = e11[k]
        strain[1] = e22[k]
        strain[2] = e33[k]
        eps[0] = strain
        sigma, ddsdde = solve_ramberg_osgood(
            E, NU, ALPHA, NEXP, SIGY, NGAUSS, gdim, eps, stress
        )
        bools.append(np.isclose(sigma[0][1], s22[k]))

    return all(bools)


"""VOIGT notation

components: t11, t22, t33, sqrt(2) t12, sqrt(2) t13, sqrt(2) t23
tensor basis (x denotes dyad product):
    e_V1 = e1 x e1,
    e_V2 = e2 x e2,
    e_V3 = e3 x e3,
    e_V4 = (e2 x e1 + e1 x e2) / sqrt(2),
    e_V5 = (e1 x e3 + e3 x e1) / sqrt(2),
    e_V6 = (e2 x e3 + e3 x e2) / sqrt(2),

This leads to the tetrad as in Betram & Gluege
'Solid Mechanics' page 101.

in 2D:
    components: t11, t22, t33, sqrt(2) t12
    with e.g. t33 = 0 in plane strain conditions
"""


def eps(v):
    gdim = v.geometric_dimension()
    e = sym(grad(v))
    if gdim == 2:
        return as_vector([e[0, 0], e[1, 1], 0.0, 2 ** 0.5 * e[0, 1]])
    elif gdim == 3:
        return as_vector(
            [
                e[0, 0],
                e[1, 1],
                e[2, 2],
                2 ** 0.5 * e[0, 1],
                2 ** 0.5 * e[0, 2],
                2 ** 0.5 * e[1, 2],
            ]
        )
    else:
        assert False




class RambergOsgoodProblem(NonlinearProblem):
    def __init__(self, mesh, deg_d, deg_q, **kwargs):
        NonlinearProblem.__init__(self)

        metadata = {"quadrature_degree": deg_q, "quadrature_scheme": "default"}
        dxm = dx(metadata=metadata)

        cell = mesh.ufl_cell()
        self.gdim = mesh.geometric_dimension()

        # solution field
        Ed = VectorElement("CG", cell, degree=deg_d)

        self.V = FunctionSpace(mesh, Ed)
        self.d = Function(self.V, name="displacement")

        # generic quadrature function spaces
        q = "Quadrature"
        voigt = self.gdim * 2  # 4 or 6
        QF = FiniteElement(q, cell, deg_q, quad_scheme="default")
        QV = VectorElement(q, cell, deg_q, quad_scheme="default", dim=voigt)
        QT = TensorElement(q, cell, deg_q, quad_scheme="default", shape=(voigt, voigt))
        VQF, VQV, VQT = [FunctionSpace(mesh, Q) for Q in [QF, QV, QT]]

        self.NGAUSS = VQF.dim()

        # quadrature function
        self.q_sigma = Function(VQV, name="current stresses")
        self.q_eps = Function(VQV, name="current strains")
        self.q_dsigma_deps = Function(VQT, name="stress-strain tangent")

        dd = TrialFunction(self.V)
        d_ = TestFunction(self.V)

        # int eps : C : eps dx - f_ext + int eps : sigma dx == 0 is expected?
        # dR + R - f_ext == 0
        # need to subtract external forces later

        self.R = inner(eps(d_), self.q_sigma) * dxm
        self.dR = inner(eps(dd), self.q_dsigma_deps * eps(d_)) * dxm

        self.calculate_eps = LocalProjector(eps(self.d), VQV, dxm)

        self.assembler = None

    def evaluate_material(self):
        # project the strain onto their quadrature spaces and ...
        self.calculate_eps(self.q_eps)
        strain = self.q_eps.vector().get_local()
        stress = self.q_sigma.vector().get_local()

        # TODO improve handling of material parameters ...
        E = 210e3
        NU = 0.3
        ALPHA = 0.01
        NEXP = 5.0
        SIGY = 500.0

        # ... "manually" evaluate_material the material ...
        sigma, ddsdde = solve_ramberg_osgood(
            E,
            NU,
            ALPHA,
            NEXP,
            SIGY,
            self.NGAUSS,
            self.gdim,
            strain.reshape(self.NGAUSS, self.gdim * 2),
            stress.reshape(self.NGAUSS, self.gdim * 2),
        )

        # ... and write the calculated values into their quadrature spaces.
        set_q(self.q_sigma, sigma)
        set_q(self.q_dsigma_deps, ddsdde)

    def update(self):
        # not needed for Ramberg Osgood
        pass

    def set_bcs(self, bcs):
        # Only now (with the bcs) can we initialize the assembler
        self.assembler = SystemAssembler(self.dR, self.R, bcs)

    def F(self, b, x):
        if not self.assembler:
            raise RuntimeError("You need to `.set_bcs(bcs)` before the solve!")
        self.evaluate_material()
        self.assembler.assemble(b, x)

    def J(self, A, x):
        self.assembler.assemble(A)


def get_neumann(dim, force):
    f = df.Expression(("0.0", "F * time"), degree=0, F=force, time=0.0, name="f")

    class Top(df.SubDomain):
        tol = 1e-6

        def inside(self, x, on_boundary):
            return on_boundary and df.near(x[1], 1.0, self.tol)

    neumann = Top()
    return f, neumann


def get_dirichlet(dim, V):
    bcs = []

    class Bottom(df.SubDomain):
        tol = 1e-6

        def inside(self, x, on_boundary):
            return on_boundary and df.near(x[1], 0.0, self.tol)

    origin = df.CompiledSubDomain("near(x[0], 0.0) && near(x[1], 0.0)")
    bcs.append(df.DirichletBC(V.sub(1), df.Constant(0.0), Bottom()))
    bcs.append(df.DirichletBC(V, df.Constant((0.0, 0.0)), origin, method="pointwise"))
    return bcs


def simple_tension(args, mesh, matparam):
    """
    simple tension test
    """

    gdim = mesh.geometric_dimension()
    Q = FiniteElement("Quadrature", mesh.ufl_cell(), 1, quad_scheme="default")
    VQ = FunctionSpace(mesh, Q)
    ngauss = VQ.dim()

    ro = RambergOsgoodProblem(mesh, args["ORDER"], args["QDEG"])

    facets = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    ds = Measure("ds")(subdomain_data=facets)
    facets.set_all(0)

    # external load
    max_load = args["--load"]
    traction, neumann = get_neumann(gdim, max_load)
    neumann.mark(facets, 99)
    d_ = TestFunction(ro.V)
    force = dot(traction, d_) * ds(99)
    ro.R -= force

    # dirichlet bcs
    bcs = get_dirichlet(gdim, ro.V)
    ro.set_bcs(bcs)

    solver = NewtonSolver()
    solver.parameters["linear_solver"] = "mumps"
    solver.parameters["maximum_iterations"] = 10
    solver.parameters["error_on_nonconvergence"] = False

    x_at_top = (0.5, 1.0)

    nTime = args["--steps"]
    load_steps = np.linspace(0, 1, num=nTime + 1)[1:]
    abs_time = 0
    iterations = np.array([], dtype=np.int)
    displacement = [0.0, ]
    load = [0.0, ]

    for (inc, time) in enumerate(load_steps):
        print("Load Increment:", inc)
        traction.time = time
        niter, converged = solver.solve(ro, ro.d.vector())
        assert converged
        iterations = np.append(iterations, niter)

        # load displacement data
        displacement.append(ro.d(x_at_top)[1])
        load.append(traction(x_at_top)[1])

    # ### analytical solution
    displacement = np.array(displacement)
    load = np.array(load)
    sol = SimpleTensionSolution2D(args["--load"], **matparam)
    e11, e22, s22 = sol.solve()
    w = sol.energy()
    I = np.trapz(load, displacement)

    if args["--plot"]:
        fig, ax = plt.subplots()
        ax.plot(e22, s22, "r-", label="analytical")
        ax.plot(displacement, load, "bo", label="num")
        ax.set_xlabel(r"$\varepsilon_{yy}$")
        ax.set_ylabel(r"$\sigma_{yy}$")
        ax.legend()
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        plt.show()

    if args["--write"]:
        results = root / "results"
        np.save(results / f"numpy_jitted_stress_strain_{args['N']}.npy", np.column_stack((displacement, load)))
        np.save(results / f"numpy_jitted_iterations_{args['N']}.npy", iterations)

    return np.isclose((w - I) / w, 0.0, atol=1e-2)


if __name__ == "__main__":
    main(sys.argv[1:])
