"""
Ramberg Osgood material law
===========================

Introduction and governing equations
------------------------------------
The ramberg osgood material law can be used to model ductile behaviour for
monotonic loading and is often used in fracture mechanics applications. In
contrast to incremental plasticity models stress and strain are directly
related and thus the ramberg osgood model is in fact a nonlinear elastic model.
While algorithmically the solution of the ramberg osgood constitutive law
in a FE code is rather simple, it involves the solution of a power law on
integration point level which can be challenging with regard to the
implementation in FEniCSx, UFL respectively.
As in the other examples in the following we subclass
``dolfinx.fem.petsc.NonlinearProblem`` to interact with
``dolfinx.nls.petsc.NewtonSolver`` and solve the linearized principle of vitual
power in each iteration. The consistent tangent and stress are functions in a
quadrature space and _filled_ manually after solving the constitutive equations
in a pure numpy code.

Linearized principle of virtual power:

.. math::
    \int_\Omega \bm \varepsilon \cdot
    \frac{\partial \bm\sigma}{\partial \bm\varepsilon} \cdot \bm \varepsilon \;\mathrm{d}x
    = f_{\mathrm{ext}} - \int_\Omega \bm \sigma \cdot \bm \varepsilon \;\mathrm{d}x


Constitutive law
****************
For the sake of brevity we skip a derivation of the equations and only summarize
the ones essential for the presented implementation.
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

Equivalent stress and equivalent strain are related via a power law and for given
$\varepsilon_{\mathrm{v}}$ we can determine $\sigma_{\mathrm{v}}$ by finding the
root of:

.. math::
    f(\sigma_{\mathrm{v}}) = \frac{2}{3} \sigma_{\mathrm{v}} \left(
    \frac{1}{2G} + \frac{3 \alpha}{2E} \left(\frac{\sigma_{\mathrm{v}}}{\sigma_{\mathrm{y}}}\right)^{n-1}
    \right) - \varepsilon_{\mathrm{v}}\,.

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

Algorithm to compute stress and consistent tangent for a given strain state:
    1. Compute equivalent strain $\varepsilon_{\mathrm{v}}$,
    2. Compute equivalent stress $\sigma_{\mathrm{v}}$ via newton method (previous stress state can be used as initial guess),
    3. Compute stress,
    4. Compute consistent tangent

"""

"""
Implementation
==============

Imports
-------
We start by importing the usual ``dolfinx`` modules and other packages.

"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from typing import Callable
from mpi4py import MPI
import numpy as np
import dolfinx as df
import ufl

from fenics_constitutive.interfaces import (
    Constraint,
    IncrSmallStrainModel,
    IncrSmallStrainProblem,
)
from fenics_constitutive.stress_strain import ufl_mandel_strain

"""

Solution of the constitutive law
--------------------------------

For the solution of the Ramberg Osgood material law, we use a python
class that holds the material parameters and implements a
method `evaluate` that follows the interface defined in
`fenics_constitutive.interfaces.IncrSmallStrainModel.evaluate`.
Given the current strain, the method computes the current stress and consistent
tangent.
The solution of the power law mentioned above makes a vectorization of the numpy
code difficult. Hence we could use a C++ function/class to solve the
constitutive law. Another option is the use of
`numba <https://numba.pydata.org/>`_ to speed up the numpy code.

"""


class RambergOsgood3D(IncrSmallStrainModel):
    gdim = 3

    def __init__(self, param: dict[str, float]):
        # material parameters, constants
        self.e = param["e"]
        self.nu = param["nu"]
        self.alpha = param["alpha"]
        self.n = param["n"]
        self.sigy = param["sigy"]
        self.k = self.e / (1.0 - 2.0 * self.nu)
        self.g = self.e / 2.0 / (1.0 + self.nu)
        self.lam = self.e * self.nu / (1 + self.nu) / (1 - 2 * self.nu)
        self.mu = self.e / (2 * (1 + self.nu))
        self.Cel = np.array(  # elastic tangent
            [
                [self.lam + 2 * self.mu, self.lam, self.lam, 0.0, 0.0, 0.0],
                [self.lam, self.lam + 2 * self.mu, self.lam, 0.0, 0.0, 0.0],
                [self.lam, self.lam, self.lam + 2 * self.mu, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2 * self.mu, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 2 * self.mu, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 2 * self.mu],
            ]
        )
        self.zero_strain_tolerance = 1e-12
        self.sv_tol = 1e-12
        self.maxiter = 50
        # helpers voigt notation
        self.I2 = np.zeros(
            self.stress_strain_dim, dtype=np.float64
        )  # Identity of rank 2 tensor
        self.I2[0] = 1.0
        self.I2[1] = 1.0
        self.I2[2] = 1.0
        self.I4 = np.eye(
            self.stress_strain_dim, dtype=np.float64
        )  # Identity of rank 4 tensor

    def evaluate(
        self,
        del_t: float,
        grad_del_u: np.ndarray,
        mandel_stress: np.ndarray,
        tangent: np.ndarray,
        history: np.ndarray | dict[str, np.ndarray],
    ) -> None:
        stress_view = mandel_stress.reshape(-1, self.stress_strain_dim)
        tangent_view = tangent.reshape(-1, self.stress_strain_dim**2)

        for n, eps in enumerate(grad_del_u.reshape(-1, self.stress_strain_dim)):
            # eps = strain at time t + delta t
            tr_eps = np.sum(eps[:3])
            eps_dev = eps - tr_eps * self.I2 / 3
            ev = np.sqrt(2.0 / 3.0 * np.dot(eps_dev, eps_dev))

            if ev < self.zero_strain_tolerance:
                # return elastic tangent
                stress_view[n] = self.Cel @ eps
                tangent_view[n] = self.Cel.flatten()
            else:
                # compute correct tangent and stress
                # stress at time t
                sig = stress_view[n]
                tr_sig = np.sum(sig[:3])
                sig_dev = sig - tr_sig * self.I2 / 3
                # equivalent stress at time t is used as initial guess
                sv_initial = np.sqrt(3.0 / 2.0 * np.dot(sig_dev, sig_dev))

                # stress at time t + delta t
                if sv_initial <= self.sigy:
                    sv = sv_initial
                else:
                    # initial_guess is > sigy
                    sv = (self.sigy ** (self.n - 1.0) * self.e * ev / self.alpha) ** (
                        1.0 / self.n
                    )

                def f(x):
                    stuff = 1.0 / (2.0 * self.mu) + 3.0 / (
                        2.0 * self.e
                    ) * self.alpha * (x / self.sigy) ** (self.n - 1.0)
                    return stuff * 2.0 / 3.0 * x - ev

                def df(x):
                    return 1.0 / (3.0 * self.mu) + self.n * self.alpha / self.e * (
                        x / self.sigy
                    ) ** (self.n - 1.0)

                s = f(sv)
                ds = df(sv)

                niter = 0
                while abs(f(sv)) > self.sv_tol:
                    sv = sv - s / ds
                    s = f(sv)
                    ds = df(sv)
                    niter += 1
                    if niter > self.maxiter:
                        break

                sig_dev = 2.0 * sv / 3.0 / ev * eps_dev
                tr_sig = self.k * tr_eps
                sig = tr_sig * self.I2 / 3.0 + sig_dev
                stress_view[n] = sig

                nenner = sv / (
                    3.0 * self.mu
                ) + self.alpha * self.n * self.sigy / self.e * (
                    (sv / self.sigy) ** (self.n)
                )
                tangent = 2 * sv / 3 / ev * (
                    self.I4
                    - 2.0
                    / 3.0
                    / ev
                    * (1.0 / ev - 1.0 / nenner)
                    * np.outer(eps_dev, eps_dev)
                ) + 1.0 / 3.0 * (self.k - 2 * sv / (3 * ev)) * np.outer(
                    self.I2, self.I2
                )
                tangent_view[n] = tangent.flatten()

    def update(self) -> None:
        pass

    @property
    def constraint(self) -> Constraint:
        return Constraint.FULL

    @property
    def history_dim(self) -> int:
        return 0


"""

Mechanics Problem
-----------------

We subclass ``dolfinx.fem.petsc.NonlinearProblem`` to define our _ramberg
osgood problem_. This will allow us to make use of the implementation of
Newton's method in FEniCSx via ``dolfinx.nls.petsc.NewtonSolver``.
We then need to make sure, that the weak form is implemented correctly
and that the stress and consistent tangent are updated in each iteration
until equilibrium is reached. To this end, we override the ``form`` method
of ``dolfinx.fem.petsc.NonlinearProblem`` and after updating the current
solution call the materials' ``evaluate`` method to update the stress state
as well.

"""


def create_meshtags(
    domain: df.mesh.Mesh, entity_dim: int, markers: dict[str, tuple[int, Callable]]
) -> tuple[df.mesh.MeshTagsMetaClass, dict[str, int]]:
    """Creates meshtags for the given markers.

    This code is part of the FEniCSx tutorial
    by Jørgen S. Dokken.
    See https://jsdokken.com/dolfinx-tutorial/chapter3/robin_neumann_dirichlet.html?highlight=sorted_facets#implementation # noqa: E501

    Args:
        domain: The computational domain.
        entity_dim: Dimension of the entities to mark.
        markers: The definition of subdomains or boundaries where each key is a string
          and each value is a tuple of an integer and a marker function.

    """
    tdim = domain.topology.dim
    assert entity_dim in (tdim, tdim - 1)

    entity_indices, entity_markers = [], []
    edim = entity_dim
    marked = {}
    for key, (marker, locator) in markers.items():
        entities = df.mesh.locate_entities(domain, edim, locator)
        entity_indices.append(entities)
        entity_markers.append(np.full_like(entities, marker))
        if entities.size > 0:
            marked[key] = marker
    entity_indices = np.hstack(entity_indices).astype(np.int32)
    entity_markers = np.hstack(entity_markers).astype(np.int32)
    sorted_facets = np.argsort(entity_indices)
    mesh_tags = df.mesh.meshtags(
        domain, edim, entity_indices[sorted_facets], entity_markers[sorted_facets]
    )
    return mesh_tags, marked


class RambergOsgoodProblem(IncrSmallStrainProblem):
    def __init__(self, laws, u, q_degree: int = 2):
        super().__init__(laws, u, q_degree=q_degree)

    def eps(self, u):
        constraint = self.constraint
        return ufl_mandel_strain(u, constraint)


"""

Voigt notation
--------------

It is common practice in computational mechanics to store only six
of the nine components of the symmetric (cauchy) stress and strain tensors.
We choose an orthonormal tensor (voigt) basis which preserves the properties of
the scalar product, hence the $\sqrt{2}$ below.
For more information see e.g. the book
`Festkörpermechanik (Solid Mechanics), by Albrecht Bertram and Rainer Glüge, <https://opendata.uni-halle.de/bitstream/1981185920/11636/1/Bertram%20Gl%C3%BCge_Festk%C3%B6rpermechanik%202013.pdf>`_
which is available (in german) online.

"""

"""
Example
========

Simple Tension Test
-------------------

To test the above implementation we compare our numerical results to the
analytical solution for a (simple) tension test in 3D, where the Cauchy stress
is given as

.. math::
    \boldsymbol{\sigma} = \sigma \boldsymbol{e}_1 \otimes\boldsymbol{e}_1.

"""


class RambergOsgoodSimpleTension:
    def __init__(self, param: dict[str, float]):
        self.e = param["e"]
        self.nu = param["nu"]
        self.alpha = param["alpha"]
        self.n = param["n"]
        self.sigy = param["sigy"]
        self.k = self.e / (1.0 - 2.0 * self.nu)
        self.g = self.e / 2.0 / (1.0 + self.nu)

    def energy(self):
        assert np.sum(self.sigma) > 0.0
        return np.trapz(self.sigma, self.eps)

    def solve(self, max_load, num_points=21) -> None:
        self.sigma = np.linspace(0, max_load, num=num_points)

        E = self.e
        ALPHA = self.alpha
        N = self.n
        SIGY = self.sigy
        K = self.k
        G = self.g

        self.eps = self.sigma / 3.0 / K + 2 / 3.0 * self.sigma * (
            1.0 / 2.0 / G + 3 * ALPHA / 2.0 / E * (self.sigma / SIGY) ** (N - 1)
        )
        # eps_22 = sigma / 3.0 / K - 1 / 3.0 * sigma * (
        #     1.0 / 2.0 / G + 3 * ALPHA / 2.0 / E * (sigma / SIGY) ** (N - 1)
        # )
        # eps_33 = sigma / 3.0 / K - 1 / 3.0 * sigma * (
        #     1.0 / 2.0 / G + 3 * ALPHA / 2.0 / E * (sigma / SIGY) ** (N - 1)
        # )


# The function to run the force-controlled simple tension test.
# We apply a constant force on the right surface, pulling in the
# :math:`\boldsymbol{e}_1`-direction.
def simple_tension_test(mesh, material, pltshow=False):
    function_space = df.fem.VectorFunctionSpace(mesh, ("Lagrange", 1))
    u = df.fem.Function(function_space)

    # ### Definition of BCs for Simple Tension Test
    def origin(x):
        p = [0.0, 0.0, 0.0]
        return np.logical_and(
            np.logical_and(np.isclose(x[0], p[0]), np.isclose(x[1], p[1])),
            np.isclose(x[2], p[2]),
        )

    def x_001(x):
        p = [0.0, 0.0, 1.0]
        return np.logical_and(
            np.logical_and(np.isclose(x[0], p[0]), np.isclose(x[1], p[1])),
            np.isclose(x[2], p[2]),
        )

    def left(x):
        return np.isclose(x[0], 0.0)

    tdim = mesh.topology.dim
    fdim = tdim - 1

    origin_vertex = df.mesh.locate_entities_boundary(mesh, 0, origin)
    x3_vertex = df.mesh.locate_entities_boundary(mesh, 0, x_001)
    left_facets = df.mesh.locate_entities_boundary(mesh, fdim, left)

    # ### Dirichlet BCs
    zero_scalar = df.fem.Constant(mesh, 0.0)
    fix_ux = df.fem.dirichletbc(
        zero_scalar,
        df.fem.locate_dofs_topological(function_space.sub(0), fdim, left_facets),
        function_space.sub(0),
    )
    fix_uy = df.fem.dirichletbc(
        zero_scalar,
        df.fem.locate_dofs_topological(function_space.sub(1), 0, origin_vertex),
        function_space.sub(1),
    )
    fix_uz = df.fem.dirichletbc(
        zero_scalar,
        df.fem.locate_dofs_topological(function_space.sub(2), 0, origin_vertex),
        function_space.sub(2),
    )
    # rotation around x1-axis
    fix_rot_x1 = df.fem.dirichletbc(
        zero_scalar,
        df.fem.locate_dofs_topological(function_space.sub(1), 0, x3_vertex),
        function_space.sub(1),
    )

    # ### Neumann BCs
    def right(x):
        return np.isclose(x[0], 1.0)

    neumann_tag = 15
    neumann_boundary = {"right": (neumann_tag, right)}
    facet_tags, _ = create_meshtags(mesh, fdim, neumann_boundary)
    dA = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    max_load = 2718.0
    neumann_data = df.fem.Constant(mesh, (max_load, 0.0, 0.0))

    laws = [(material, None)]
    problem = RambergOsgoodProblem(laws, u)
    # neumann
    test_function = ufl.TestFunction(u.function_space)
    fext = ufl.inner(neumann_data, test_function) * dA(neumann_tag)
    problem.R_form -= fext

    # dirichlet
    dirichlet = [fix_ux, fix_uy, fix_uz, fix_rot_x1]
    problem.compile(dirichlet)  # optionally add form compiler options

    solver = df.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"

    # ### center point, right surface
    xc_right = np.array([[1.0, 0.5, 0.5]])

    nTime = 10
    load_steps = np.linspace(0, 1, num=nTime + 1)[1:]
    iterations = np.array([], dtype=np.int32)
    displacement = [0.0]
    load = [0.0]

    for inc, time in enumerate(load_steps):
        print("Load Increment:", inc)

        # external force
        current_load = time * max_load
        neumann_data.value = (current_load, 0.0, 0.0)

        niter, converged = solver.solve(u)
        assert converged
        print(f"Converged: {converged} in {niter} iterations.")
        iterations = np.append(iterations, niter)

        # load displacement data
        u_right = u.eval(xc_right, cells=problem.cells)
        displacement.append(u_right.item(0))
        load.append(current_load)

    displacement = np.array(displacement)
    load = np.array(load)
    return load, displacement


# main function to run the simple tension test.
def main(args):
    n = args.num_cells
    mesh = df.mesh.create_unit_cube(
        MPI.COMM_WORLD, n, n, n, df.mesh.CellType.hexahedron
    )
    matparam = {
        "e": 210e3,
        "nu": 0.3,
        "alpha": 0.01,
        "n": 5,
        "sigy": 500.0,
    }
    material = RambergOsgood3D(matparam)
    sigma_h, eps_h = simple_tension_test(mesh, material)

    # ### Comparison with analytical solution
    sol = RambergOsgoodSimpleTension(matparam)
    sol.solve(sigma_h[-1], num_points=51)
    w = sol.energy()
    I = np.trapz(sigma_h, eps_h)
    assert np.isclose((w - I) / w, 0.0, atol=1e-2)

    if args.show:
        ax = plt.subplots()[1]
        ax.plot(sol.eps, sol.sigma, "r-", label="analytical")
        ax.plot(eps_h, sigma_h, "bo", label="num")
        ax.set_xlabel(r"$\varepsilon_{xx}$")
        ax.set_ylabel(r"$\sigma_{xx}$")
        ax.legend()
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        plt.show()


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "num_cells",
        type=int,
        help="Number of cells in each spatial direction of the unit cube.",
    )
    parser.add_argument("--show", action="store_true", help="Show plot.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
