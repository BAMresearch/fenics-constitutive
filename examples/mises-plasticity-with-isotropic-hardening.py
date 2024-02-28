#Von Mises plasticity with non-linear isotropic hardening
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

class VonMises3D(IncrSmallStrainModel):
    gdim = 3

    def __init__(self, param: dict[str, float]):
        # Define 1o1 and fourth order identity tensor
        self.xioi = np.array([[1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0]])

        self.xii = np.array([[1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0.5, 0, 0],
                        [0, 0, 0, 0, 0.5, 0],
                        [0, 0, 0, 0, 0, 0.5]])
        # define deviatoric projection tensor
        self.xpp = self.xii - (1 / 3) * self.xioi

        self.I2 = np.zeros(
            self.stress_strain_dim, dtype=np.float64
        )  # Identity of rank 2 tensor
        self.I2[0] = 1.0
        self.I2[1] = 1.0
        self.I2[2] = 1.0
        self.I4 = np.eye(
            self.stress_strain_dim, dtype=np.float64
        )  # Identity of rank 4 tensor

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
        eps_n = history["eps_n"]
        eps_n = eps_n.reshape(-1, self.stress_strain_dim)
        stress_view = mandel_stress.reshape(-1, self.stress_strain_dim)
        tangent_view = tangent.reshape(-1, self.stress_strain_dim**2)

        for n, eps in enumerate(grad_del_u.reshape(-1, self.stress_strain_dim)):
            # eps = strain at time t + delta t
            # trace of strain
            tr_eps = np.sum(eps[:3])
            # deviatoric strain
            eps_dev = eps - tr_eps * self.I2 / 3
            # deviatoric trial internal forces
            sigtr = 2 * p_mu * (eps_dev - eps_n[n])
            # norm of deviatoric trial internal forces
            sigtrn = np.sqrt(np.dot(sigtr, sigtr))
            # trial yield criterion
            phitr = sigtrn - np.sqrt(2 / 3) * (self.p_y0 + (self.p_y00 - self.p_y0) * (1 - np.exp(-self.p_w * history["alpha"])))
            # CHECK YIELD CRITERION
            # elastic - plastic step
            if phitr > 0:
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
                    return sigtrn - 2 * self.p_mu * x - np.sqrt(2 / 3) * (
                                self.p_y0 + (self.p_y00 - self.p_y0) * (1 - np.exp(-self.p_w * (history["alpha"] + np.sqrt(2 / 3) * x))))

                def df(x):
                    return - 2 * self.p_mu - (2 / 3) * (self.p_y00 - self.p_y0) * self.p_w * np.exp(-self.p_w * (history["alpha"] + np.sqrt(2 / 3) * x))

                s = f(sv)
                ds = df(sv)
                # start Newton iteration
                while np.abs(xr) > tol:
                    it = it + 1
                    # compute residium
                    xr = f(gamma)
                    # compute tangent
                    xg = df(gamma)
                    # update plastic flow
                    gamma = gamma - xr / xg
                    # exit Newton algorithm for iteration > nmax
                    if it > nmax:
                        print('No Convergence in Newton Raphson Iteration')
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
            hnew_eps_n = history["eps_n"] + gamma * xn
            hnew_alpha = history["alpha"] + np.sqrt(2 / 3) * gamma

            # determine elastic-plastic stresses
            sh =  self.p_ka * tr_eps * self.I2 + sigtr - 2 * p_mu * gamma * xn

            stress_view[n] = sh

            # determine elastic-plastic moduli
            aah = self.p_ka * xioi + 2 * p_mu * (1 - 2 * p_mu * xc2) * xpp + 4 * p_mu * p_mu * (xc2 - xc1) * np.outer(xn,xn)

            tangent_view[n] = aah.flatten()

    def update(self) -> None:
        pass

    @property
    def constraint(self) -> Constraint:
        return Constraint.FULL

    @property
    def history_dim(self) -> int:
        return 0


