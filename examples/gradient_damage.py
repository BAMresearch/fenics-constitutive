"""
.. _gdm-label:

Implicit gradient-enhanced damage model
=======================================

Introduction and governing equations
------------------------------------


A local isotropic damage model...

.. math::

    \nabla \cdot \left( (1-\omega(\kappa(\| \bm \varepsilon \|))) 
        \bm C : \bm \varepsilon \right)  = \bm 0

... is basically the modified linear-elastic momentum balance equation with

    - elasticity tensor $\bm C$ - build from $E, \nu$ or Lamé constants
      $\lambda, \mu$
    - damage $\omega \in [0,1)$ that is driven by the history variable $\kappa$
    - strains $\bm \varepsilon = \frac{1}{2}(\nabla \bm d + (\nabla \bm d)^T)$, 
      here in Voigt notation, meaning a vector of 
      $[\bm \varepsilon_{xx}, \bm \varepsilon_{yy}, \frac{1}{2}\bm \varepsilon_{xy}]^T$
    - the KKT conditions for $\kappa$ translate to 
      $\kappa = \max(\kappa, \|\bm \varepsilon\|)$, where the latter term is a 
      scalar norm of the *local* strains

This *local* model exhibits various numerical problems, e.g. localization in 
single bands of elements with a vanishing fracture energy upon mesh refinement. 
One way to overcome these issues is to use nonlocal models that introduce a
mesh-independent length scale. Here, this is by replacing the *local* strain
norm with the *nonlocal equivalent strains* $\bar \varepsilon$ as an additional
degree of freedom (DOF) that is calculated by a screened Poisson equation that
limits its curvature. Now, the full model reads

.. math::

    \nabla \cdot \left( (1-\omega(\kappa(\bar \varepsilon))) \bm C : \bm \varepsilon \right)  &= \bm 0  \\
    \bar \varepsilon - l^2 \Delta \bar \varepsilon &= \| \bm \varepsilon \|

and details can be found, e.g.,  in

    - the original paper `Gradient enhanced damage for quasi‐brittle materials, Peerlings et al., 1996 <https://doi.org/10.1002/(SICI)1097-0207(19961015)39:19\<3391::AID-NME7\>3.0.CO;2-D>`_ and
    - a more recent paper discussing alternative solution strategies `Implicit–explicit integration of gradient-enhanced damage models, Titscher et al., 2019 <https://doi.org/10.1061/(ASCE)EM.1943-7889.0001608>`_

Next, we will discuss the building blocks of the constitutive model with the code.

**Remark: The functions in this section are written to work on multiple values
simultaneously. So instead of evaluating a function once per scalar, we group 
the scalars in a (possibly huge) vector and apply the function once. This may
explain some strange syntax and slicing in the code below.**

.. include:: gdm_constitutive.rst

"""

from gdm_constitutive import *
from helper import *
from dolfin.cpp.log import log

"""
Quadrature space formulation
----------------------------

Degrees of freedom in a mixed function space
    * u = [d, e] 
    * u = total mixed vector
    * d = displacement field $\bm d$
    * e = nonlocal equivalent strain field $\bar \varepsilon$


Momentum balance + Screened Poisson ...
::

    Rd = eps(dd) : sigma(eps(d), e) * dx
    Re = de * e * dx + grad(de) . l ** 2 * grad(e) * dx  - de * eeq(eps) * dx

plus their derivatives
::

    dRd/dd = eps(dd) : (dSigma_deps) * eps(d)) * dx
    dRd/de = de * (dSigma_de * eps(d)) *dx
    dRe/dd = eps(dd) * (-deeq_deps) * e * dx
    dRe/de = de * e * dx + grad(de) . l**2 * grad(e) * dx

The *trivial* terms in the equations above are implemented using FEniCS
forms. The non-trivial ones are defined as functions with prefix ``q_`` in 
appropriately-sized quadrature function spaces. The values for those functions
are calculated in the ``GDMPlaneStrain`` class above.

"""

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
        d_, e_ = TestFunctions(self.Vu)
        d, e = split(self.u)

        try:
            f_d = kwargs["f_d"]
        except:
            f_d = Constant(1.0)

        eps = self.mat.eps
        self.R = f_d * inner(eps(d_), self.q_sigma) * dxm
        self.R += e_ * (e - self.q_eeq) * dxm
        self.R += dot(grad(e_), mat.l ** 2 * grad(e)) * dxm

        self.dR = f_d * inner(eps(d_), self.q_dsigma_deps * eps(dd)) * dxm
        self.dR += f_d * inner(eps(d_),  self.q_dsigma_de * de) * dxm
        self.dR += e_ * (de - dot(self.q_deeq_deps, eps(dd))) * dxm
        #∂grad(e)/∂e de = linear terms of grad(e+de) = grad(de)
        self.dR += dot(grad(e_), mat.l ** 2 * (grad(de))) * dxm

        self.calculate_eps = LocalProjector(eps(d), VQV, dxm)
        self.calculate_e = LocalProjector(e, VQF, dxm)

        self.assembler = None

    def evaluate_material(self):
        # project the strain and the nonlocal equivalent strains onto
        # their quadrature spaces and ...
        self.calculate_eps(self.q_eps)
        self.calculate_e(self.q_e)

        eps_flat = self.q_eps.vector().get_local()
        e = self.q_e.vector().get_local()

        # ... "manually" evaluate_material the material ...
        self.mat.evaluate(eps_flat, e)

        # ... and write the calculated values into their quadrature spaces.
        set_q(self.q_eeq, self.mat.eeq)
        set_q(self.q_deeq_deps, self.mat.deeq)
        set_q(self.q_sigma, self.mat.sigma)
        set_q(self.q_dsigma_deps, self.mat.dsigma_deps)
        set_q(self.q_dsigma_de, self.mat.dsigma_de)

    def update(self):
        self.calculate_e(self.q_e)
        self.mat.update(self.q_e.vector().get_local())
        set_q(self.q_k, self.mat.kappa) # just for post processing

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

"""
**Remark:** Subclassing from ``dolfin.NonlinearProblem`` ...
    * ... is rather straight forward as we only need to pass our forms and boundary
      conditions to the ``dolfin.SystemAssembler`` and call it in the 
      overwritten methods ``F`` (assembles the out-of-balance forces) and ``J`` 
      (assembles the tangent) ...
    * ... and allows us to directly use ``dolfin.NewtonSolver`` - the Newton-Raphson
      implementation of FEniCS. Note that this algorithm (as probably every NR)
      always evaluates ``F`` before ``J``. So it is sufficient to perform 
      the ``GDM.evaluate_material`` only in ``F``.

Examples
--------

Comparison with the analytic solution
*************************************

The Peerlings paper cited above includes a discussion on an analytic solution 
of the model for the ``damage_perfect`` damage law. A 1D bar of length $L$
(here modeled as a thin 2D structure) has a cross section reduction $\alpha$ 
over the length of $W$ and is loaded by a displacement BC $\Delta L$.

Three regions form: 
    * damage in the weakened cross section
    * damage in the unweakened cross section
    * no damage

and the authors provide a solution of the PDE system for each of the regions.
Finding the remaining integration constants is left to the reader. Here, it
is solved using ``sympy``. We also subclass from ``dolfin.UserExpression`` to
interpolate the analytic solution into function spaces or calculate error norms.

.. include:: gdm_analytic.rst

"""

from gdm_analytic import PeerlingsAnalytic

class PeerlingsAnalyticExpr(UserExpression, PeerlingsAnalytic):
    def __init__(self, **kwargs):
        PeerlingsAnalytic.__init__(self)
        UserExpression.__init__(self, **kwargs)

    def eval(self, value, x):
        value[0] = self.e(x[0])

"""
Using this, we can rebuild the example with our ``GDM`` nonlinear problem and
compare.
"""

def gdm_error(n_elements):
    """
    ... evaluated in 2D
    """
    e = PeerlingsAnalyticExpr(degree=4)

    mesh = RectangleMesh(Point(0.0, 0.0), Point(e.L / 2.0, 1), n_elements, 1)
    mat = GDMPlaneStrain(E=e.E, nu=0.0, ft=e.E * e.kappa0, l=e.l, dmg=damage_perfect)

    area = Expression(
        "x[0] <= W/2. ? 10.0 * (1. - a) : 10.0", W=e.W, a=e.alpha, degree=0
    )
    gdm = GDM(mesh, mat, f_d=area)

    bc_expr = Expression("t*d", degree=0, t=0, d=e.deltaL / 2)
    bc0 = DirichletBC(gdm.Vd, (0.0, 0.0), plane_at(0.0))
    bc1 = DirichletBC(gdm.Vd.sub(0), bc_expr, plane_at(e.L / 2.0))
    gdm.set_bcs([bc0, bc1])

    solver = NewtonSolver()
    solver.parameters["linear_solver"] = "mumps"
    solver.parameters["maximum_iterations"] = 10
    solver.parameters["error_on_nonconvergence"] = False

    for t in np.linspace(0.0, 1.0, 11):
        bc_expr.t = t
        assert solver.solve(gdm, gdm.u.vector())[1]

    e_fem = gdm.u.split()[1]
    return errornorm(e, e_fem)

"""
This error should converge to zero upon mesh refinement and can be used to 
determine the order of convergence of the model.

.. image:: gdm_convergence.png
"""

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
    plt.savefig("gdm_convergence.png")
    plt.show()
    print(ps)


"""
Three-point bending test
------------------------

Just a more exciting example.

.. image:: gdm_bending.gif

Note that we pass our ``GDM`` class as well as the linear solver as a parameter.
These can be modified to use an `iterative solver <gradient_damage_iterative.html>`_.
"""

def three_point_bending(problem=GDM, linear_solver=LUSolver("mumps")):
    LX = 2000
    LY = 300
    LX_load = 100

    mesh = RectangleMesh(Point(0, 0), Point(LX, LY), 100, 15)
    mat = GDMPlaneStrain()
    gdm = problem(mesh, mat)

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
    
    solver = NewtonSolver(MPI.comm_world, linear_solver, PETScFactory.instance())
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
    plot_space = FunctionSpace(mesh, "DG", 0)

    def pp(t):
        gdm.update()

        # this fixes XDMF time stamps
        import locale
        locale.setlocale(locale.LC_NUMERIC, "en_US.UTF-8")  
        fff.write(gdm.u.split()[0], t)
        fff.write(gdm.u.split()[1], t)

        # plot the damage
        q_k = gdm.q_k
        q_w = Function(q_k.function_space())
        set_q(q_w, mat.dmg(mat, q_k.vector().get_local())[0])
        w = project(q_w, plot_space)
        w.rename("w", "w")
        fff.write(w, t)

        ld(t, assemble(gdm.R))

    TimeStepper(solve, pp, gdm.u).adaptive(1.0, dt=0.1)


if __name__ == "__main__":
    assert gdm_error(200) < 1.0e-8
    convergence_test()
    three_point_bending()
    # list_timings(TimingClear.keep, [TimingType.wall])

"""
Extensions
----------

* make everything dimension independent
"""
