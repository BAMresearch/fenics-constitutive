"""
Basic principle
===============

This example demonstrates the basic principle of performing the "constitutive
update" manually on the quadrature points, by explicitly providing the stresses
and their derivative w.r.t. the strains as ``numpy`` matrices.

This is done in a displacement-controlled, uniaxial tensile test on a unit square.

    Note that this is not meant as an introduction on FEM or linear elasticity, but
    focuses on the steps on how to replace the constitutive formulation in UFL
    with a custom one.

"""

from helper import *

"""
For details on this ``helper`` functions/classes, see `here <helper.html>`_.


As with a normal fenics simulation, we first define the mesh, function spaces
and boundary conditions.
"""

mesh = UnitSquareMesh(20, 20)
V = VectorFunctionSpace(mesh, "P", 2)
du, u_ = TrialFunction(V), TestFunction(V)


u_bc = 42.0
bc_right = DirichletBC(V.sub(0), u_bc, plane_at(1, "x"))
bc_left = DirichletBC(V.sub(0), 0.0, plane_at(0, "x"))
bc_origin = DirichletBC(V.sub(1), 0.0, plane_at(0, "y"))
bcs = [bc_right, bc_left, bc_origin]

"""
Constitutive law
----------------


Instead of providing the constitutive law for linear elasticity via UFL, 
we manually implement. This is done in `Voigt notation <https://en.wikipedia.org/wiki/Voigt_notation>`_, where the
strains $\bm \varepsilon$ are defined as
"""


def eps(u):
    e = sym(grad(u))
    return as_vector((e[0, 0], e[1, 1], 2 * e[0, 1]))


"""
and `Hooke's law <https://en.wikipedia.org/wiki/Hooke%27s_law>`_ (for plane 
stress) reads 

.. math::
         \bm \sigma =  \begin{bmatrix}\sigma_{11} \\ \sigma_{22} \\ \sigma_{12} \end{bmatrix}
 \,=\, \frac{E}{1-\nu^2}
 \begin{bmatrix} 1 & \nu & 0 \\
 \nu & 1 & 0 \\
 0 & 0 & \frac{1-\nu}{2} \end{bmatrix}
 \begin{bmatrix}\varepsilon_{11} \\ \varepsilon_{22} \\ 2\varepsilon_{12} \end{bmatrix}
    = \bm C \bm \varepsilon \\
"""

E, nu = 20000, 0.2
C11 = E / (1.0 - nu * nu)
C12 = C11 * nu
C33 = C11 * 0.5 * (1.0 - nu)
C = np.array([[C11, C12, 0.0], [C12, C11, 0.0], [0.0, 0.0, C33]])

"""
Quadrature spaces
-----------------

We now define appropriately sized vector function spaces for $\bm \sigma$ and 
a tensor function space for $\frac{\bm \sigma}{\bm \varepsilon}$. 
"""

q = "Quadrature"
cell = mesh.ufl_cell()
q_dim = C.shape[0]
deg_q = 2
QV = VectorElement(q, cell, deg_q, quad_scheme="default", dim=q_dim)
QT = TensorElement(q, cell, deg_q, quad_scheme="default", shape=(q_dim, q_dim))
VQV, VQT = [FunctionSpace(mesh, Q) for Q in [QV, QT]]

q_sigma = Function(VQV, name="stresses")
q_eps = Function(VQV, name="strains")
q_dsigma_deps = Function(VQT, name="stress-strain tangent")

"""
The quadrature degree ``deg_q`` defines the number of integration/Gauss points
and a corresponding integration measure has to be defined and used in all forms.
"""

metadata = {"quadrature_degree": deg_q, "quadrature_scheme": "default"}
dxm = dx(metadata=metadata)

"""
Solution
--------

The form for the residual and its derivative w.r.t. to the displacements is 
defined as
"""

R = -inner(eps(u_), q_sigma) * dxm
dR = inner(eps(du), dot(q_dsigma_deps, eps(u_))) * dxm

"""
Right now, ``q_sigma``, ``q_eps`` and ``q_dsigma_deps`` contain only zeros. 
Our initial state with a zero displacement field $\bm u==\bm 0$ leads to 
zero strains and zero stresses. So we only need to modify the tangent matrix.

The values of the quadrature function space are arranged in possibly huge 1D 
vector where the first 9 (= 3x3 = ``q_dim*q_dim``) entries correspond to the
first Gauss point, the next 9 entries to the second and so on.

As our tangent is constant, we simply ``np.tile`` our the ``C`` defined above
for each Gauss point and assign it to the function space. (`Details on set_q <helper.html#setting-values-for-the-quadrature-space>`_)
"""

n_gauss = len(q_sigma.vector().get_local()) // q_dim
C_values = np.tile(C.flatten(), n_gauss)
set_q(q_dsigma_deps, C_values)

"""
Both forms are now assembled and solved.
"""

A, b = assemble_system(dR, R, bcs)

u = Function(V, name="displacements")
solve(A, u.vector(), b)

"""
Test
----

The solution field of this uniaxial problem should now look like

.. math::
    \begin{bmatrix}u_x \\ u_y \end{bmatrix} = \begin{bmatrix} u_{bc} x \\- \nu u_{bc}y \end{bmatrix}

A brief check verifies the correctness of our solution. 
"""

test_points = np.linspace(0, 1, 5)
for x in test_points:
    for y in test_points:
        u_fem = u((x, y))
        u_correct = x * u_bc, -nu * y * u_bc
        assert np.linalg.norm(u_fem - u_correct) < 1.0e-10

"""
Extensions
----------

Strain calculation
******************

This simple example with a linear constitutive law technically only requires
the nonzero tangent ``q_dsigma_deps`` to be defined and implemented. For 
nonlinear materials, this changes. They usually depend on the strain, which can
be calculated from the displacement field by projecting ``eps(u)`` on the 
quadrature function ``q_eps``.
"""

get_strain = LocalProjector(eps(u), VQV, dxm)
get_strain(q_eps)
strains = q_eps.vector().get_local()

"""
In a nonlinear analysis, the ``get_strain`` would then be called in every 
Newton Raphson iteration. Here, we can use it for additional checks like
"""

for strain in strains.reshape((-1, q_dim)): # reshape makes it a (n_gauss,3) matrix
    eps_x, eps_y, eps_xy = strain
    assert abs(eps_x - u_bc) < 1.e-10
    assert abs(eps_y + nu * u_bc) < 1.e-10
    assert abs(eps_xy) < 1.e-10

"""
or for calculating and plotting the corresponding stresses. **Note that a 
quadrature function space cannot be interpolated!** Thus, we need to define
a different one and project the solution in there. DG0 with one value for the
whole element is often used.
"""

stresses = strains.reshape((-1, q_dim)) @ C
set_q(q_sigma, stresses)

visu_space = VectorFunctionSpace(mesh, "DG", 0, dim=q_dim)
stress_plot = project(q_sigma, visu_space)
stress_plot.rename("sigma", "sigma")


f = XDMFFile("basics.xdmf")
f.write(u,0.)
f.write(stress_plot,0.)

"""
Embedding in a nonlinear material
*********************************

One Newton-Raphson iteration would normally consist of

1) calculate $\bm \varepsilon$ as a big ``numpy`` vector
2) calculate $\bm \sigma(\bm \varepsilon), \frac{\partial \bm \sigma}{\partial \bm \varepsilon}(\bm \varepsilon)$
3) assign those values back to their quadrature spaces
4) assemble and solve for a increment in $\bm u$

This is (e.g.) demonstrated in the more complex `gradient damage <gradient_damage.html>`_ example.

Iterations on Gauss point level
*******************************

The constitutive equation of plasticity models often includes a stress norm. 
This requires an iterative update of some internal variables, e.g. a plastic
multiplier, until Gauss-point-convergence is reached. This procedure may
include a manual loop over all Gauss points, where each evaluation itself 
consists of several Newton-Raphson iterations. The `Ramberg-Osgood <ramberg_osgood.html>`_
example demonstrates this approach and how the ``numba`` just-in-time compiler
significantly speeds up these calculations.
"""

