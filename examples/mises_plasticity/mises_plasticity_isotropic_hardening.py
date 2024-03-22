#Von Mises plasticity with non-linear isotropic hardening
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from typing import Callable
from mpi4py import MPI
import numpy as np
import dolfinx as df
import Simple_Tension_Test
import ufl

from fenics_constitutive import (
    Constraint,
    IncrSmallStrainModel,
    IncrSmallStrainProblem,
    strain_from_grad_u,
)
from fenics_constitutive.stress_strain import ufl_mandel_strain

class VonMises3D(IncrSmallStrainModel):
    def __init__(self, param: dict[str, float]):
        # Define 1o1 and fourth order identity tensor
        self.xioi = np.array([[1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0]])

        #self.xii = np.array([[1, 0, 0, 0, 0, 0],
        #               [0, 1, 0, 0, 0, 0],
        #                [0, 0, 1, 0, 0, 0],
        #                [0, 0, 0, 0.5, 0, 0],
        #                [0, 0, 0, 0, 0.5, 0],
        #                [0, 0, 0, 0, 0, 0.5]])
        # define deviatoric projection tensor


        self.I2 = np.zeros(
            self.stress_strain_dim, dtype=np.float64
        )  # Identity of rank 2 tensor
        self.I2[0] = 1.0
        self.I2[1] = 1.0
        self.I2[2] = 1.0
        self.I4 = np.eye(
            self.stress_strain_dim, dtype=np.float64
        )  # Identity of rank 4 tensor

        self.xpp = self.I4 - (1 / 3) * self.xioi

        # restore material parameters
        self.p_ka = param["p_ka"]  # kappa - bulk modulus
        self.p_mu = param["p_mu"]  # mu - shear modulus
        self.p_y0 = param["p_y0"]   # y0 - initial yield stress
        self.p_y00 = param["p_y00"]   # y00 - final yield stress
        self.p_w = param["p_w"]   # w - saturation parameter


    def evaluate(
        self,
        del_t: float,
        grad_del_u: np.ndarray,
        mandel_stress: np.ndarray,
        tangent: np.ndarray,
        history: np.ndarray | dict[str, np.ndarray],
    ) -> None:
        #print(history)
        eps_n = history["eps_n"].reshape(-1, self.stress_strain_dim)
        alpha = history["alpha"]
        stress_view = mandel_stress.reshape(-1, self.stress_strain_dim)
        tangent_view = tangent.reshape(-1, self.stress_strain_dim**2)
        del_strain = strain_from_grad_u(grad_del_u, self.constraint)

        for n, eps in enumerate(del_strain.reshape(-1, self.stress_strain_dim)):
            # eps = strain at time t + delta t
            # trace of strain
            tr_eps = np.sum(eps[:3])
            #print(eps)
            #print(tr_eps)
            # deviatoric strain
            eps_dev = eps - tr_eps * self.I2 / 3
            # deviatoric trial internal forces
            sigtr = 2 * self.p_mu * (eps_dev - eps_n[n])
            # norm of deviatoric trial internal forces
            sigtrn = np.sqrt(np.dot(sigtr, sigtr))
            #sigtrn = np.sqrt(sigtr[0] ** 2 + sigtr[1] ** 2 + sigtr[2] ** 2 +  sigtr[3] ** 2 +  sigtr[4] ** 2 +  sigtr[5] ** 2)
            # trial yield criterion
            phitr = sigtrn - np.sqrt(2 / 3) * (self.p_y0 + (self.p_y00 - self.p_y0) * (1 - np.exp(-self.p_w * alpha[n])))
            # CHECK YIELD CRITERION
            # elastic - plastic step
            if phitr > 0:
                #print('ELASTIC-PLASTIC-ELASTIC-PLASTIC-ELASTIC-PLASTIC-ELASTIC-PLASTIC')
                # initialization
                gamma = 0
                xr = 1
                it = 0
                tol = 1e-12
                nmax = 100
                # flow direction xn(i) = sigmatr(i) / sigtrn
                xn = sigtr / sigtrn
                # UPDATE PLASTIC MULTIPLIER VIA NEWTON-RAPHSON SCHEME
                def f(x):
                    return sigtrn - 2 * self.p_mu * x - np.sqrt(2 / 3) * ( self.p_y0 + (self.p_y00 - self.p_y0) * (1 - np.exp(-self.p_w * (alpha[n] + np.sqrt(2 / 3) * x))))

                def df(x):
                    return - 2 * self.p_mu - (2 / 3) * (self.p_y00 - self.p_y0) * self.p_w * np.exp(-self.p_w * (alpha[n] + np.sqrt(2 / 3) * x))

                # start Newton iteration
                while np.abs(xr) > tol:
                    it = it + 1
                    # compute residium
                    xr = f(gamma)
                    #print(gamma)
                    # compute tangent
                    xg = df(gamma)
                    # update plastic flow
                    gamma = gamma - xr / xg
                    # exit Newton algorithm for iteration > nmax
                    if it > nmax:
                        #print('No Convergence in Newton Raphson Iteration')
                        print('gamma is', xr)
                        break
                    # end of Newton iterration

                # compute tangent with converged gamma
                xg = df(gamma)

                # algorithmic parameters
                xc1 = - 1 / xg
                xc2 = gamma / sigtrn

                # ELASTIC STEP
            else:
                xn = np.zeros(6)
                gamma = 0
                xc1 = 0
                xc2 = 0

            # update eps^p_n+1 and alpha
            eps_n[n] += gamma * xn
            alpha[n] += np.sqrt(2 / 3) * gamma
            #print(eps_n[n],alpha[n])

            # determine elastic-plastic stresses
            sh =  self.p_ka * tr_eps * self.I2 + sigtr - 2 * self.p_mu * gamma * xn

            stress_view[n] = sh
            #print(sh)

            # determine elastic-plastic moduli
            aah = self.p_ka * self.xioi + 2 * self.p_mu * (1 - 2 * self.p_mu * xc2) * self.xpp + 4 * self.p_mu * self.p_mu * (xc2 - xc1) * np.outer(xn,xn)

            # aah = np.zeros([6, 6])
            # for i in range(0, 6):
            #     for j in range(0, 6):
            #         aah[i, j] = self.p_ka * self.xioi[i, j] + 2 * self.p_mu * (1 - 2 * self.p_mu * xc2) * self.xpp[i, j] + 4 * self.p_mu * self.p_mu * (
            #                     xc2 - xc1) * xn[i] * xn[j]

            tangent_view[n] = aah.flatten()

    def update(self) -> None:
        pass

    @property
    def constraint(self) -> Constraint:
        return Constraint.FULL

    @property
    def history_dim(self) -> int:
        return {"eps_n": self.constraint.stress_strain_dim(), "alpha": 1}





# class RambergOsgoodProblem(IncrSmallStrainProblem):
#     def __init__(self, laws, u, q_degree: int = 2):
#         super().__init__(laws, u, q_degree=q_degree)
#
#     def eps(self, u):
#         constraint = self.constraint
#         return ufl_mandel_strain(u, constraint)


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


# class RambergOsgoodSimpleTension:
#     def __init__(self, param: dict[str, float]):
#         self.e = param["e"]
#         self.nu = param["nu"]
#         self.alpha = param["alpha"]
#         self.n = param["n"]
#         self.sigy = param["sigy"]
#         self.k = self.e / (1.0 - 2.0 * self.nu)
#         self.g = self.e / 2.0 / (1.0 + self.nu)
#
#     def energy(self):
#         assert np.sum(self.sigma) > 0.0
#         return np.trapz(self.sigma, self.eps)
#
#     def solve(self, max_load, num_points=21) -> None:
#         self.sigma = np.linspace(0, max_load, num=num_points)
#
#         E = self.e
#         ALPHA = self.alpha
#         N = self.n
#         SIGY = self.sigy
#         K = self.k
#         G = self.g
#
#         self.eps = self.sigma / 3.0 / K + 2 / 3.0 * self.sigma * (
#             1.0 / 2.0 / G + 3 * ALPHA / 2.0 / E * (self.sigma / SIGY) ** (N - 1)
#         )
#         # eps_22 = sigma / 3.0 / K - 1 / 3.0 * sigma * (
#         #     1.0 / 2.0 / G + 3 * ALPHA / 2.0 / E * (sigma / SIGY) ** (N - 1)
#         # )
#         # eps_33 = sigma / 3.0 / K - 1 / 3.0 * sigma * (
#         #     1.0 / 2.0 / G + 3 * ALPHA / 2.0 / E * (sigma / SIGY) ** (N - 1)
#         # )
#
#
# # main function to run the simple tension test.
# def main(args):
#     n = args.num_cells
#     mesh = df.mesh.create_unit_cube(
#         MPI.COMM_WORLD, n, n, n, df.mesh.CellType.hexahedron
#     )
#     matparam = {
#         "p_ka": 175000,
#         "p_mu": 80769,
#         "p_y0": 1200,
#         "p_y00":2500,
#         "p_w": 200,
#     }
#     material = VonMises3D(matparam)
#     sigma_h, eps_h = Simple_Tension_Test.simple_tension_test(mesh, material)
#     #print(sigma_h)
#
#     # ### Comparison with analytical solution
#     # sol = RambergOsgoodSimpleTension(matparam)
#     # sol.solve(sigma_h[-1], num_points=51)
#     # w = sol.energy()
#     # I = np.trapz(sigma_h, eps_h)
#     # assert np.isclose((w - I) / w, 0.0, atol=1e-2)
#
#     if args.show:
#         ax = plt.subplots()[1]
#         #ax.plot(sol.eps, sol.sigma, "r-", label="analytical")
#         ax.plot(eps_h, sigma_h, label="numerical")
#         ax.set_xlabel(r"$\varepsilon_{xx}$")
#         ax.set_ylabel(r"$\sigma_{xx}$")
#         ax.legend()
#         ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
#         plt.show()
#
#
# if __name__ == "__main__":
#     import sys
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "num_cells",
#         type=int,
#         help="Number of cells in each spatial direction of the unit cube.",
#     )
#     parser.add_argument("--show", action="store_true", help="Show plot.")
#     args = parser.parse_args(sys.argv[1:])
#     main(args)
