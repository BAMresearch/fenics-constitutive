import numpy as np


class FenicsConstitutive:
    def __init__(self, n, dim):
        self.n = n
        self.sigma = np.zeros(n*dim)
        self.dsigma = np.zeros(n*dim*dim)

    def sigma_dsigma(self, strain, i):
        raise NotImplementedError()

    def update(strain, i):
        pass

    def evaluate(self, all_strains):
        for i in range(n):
            pass

    def define(self, ):
        pass


# integration-point based laws with no real common base class,
# as they are too diverse
class LinearElasticIP:
    def sigma(self, strain):
    def dsigma(self, strain):

class LinearElasticDamageIP:
    def sigma(self, strain, damage)

 





class LinearElasticIP:
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu
        self.D = np.r_[0.]

    def sigma(self, strain):
        return self.D * self.strain

    def dsigma(self, strain):
        return self.D 


class PlasticityIP:
    def __init__(self, E, nu, some_more):
        self.E = E
        self.nu = nu

    def sigma_dsigma(self, eps, eps_pl, p):
        pass

    def update(self):
        pass

class GDM:
    def sigma_dsigma 


