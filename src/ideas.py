import numpy as np


class Something:
    pass


class PureMechanicsInterface:
    def __init__(self, n, dim):
        self.n, self.dim = n, dim

    def evaluate(self, all_strains):
        """
        This is called by FEniCS and sets the self.sigma and self.dsigma
        members required in the FEniCS form.

        Technically, this can also be overwritten!
        """
        for ip in range(self.n):
            self.sigma[ip], self.dsigma[ip] = self.evaluate_ip(all_strains[ip], i)

    def evaluate_ip(self, ip_strain, ip):
        """
        Overwritten by each constitutive law to handle specific history data.
        Returns ip_stress and ip_strain
        """
        raise NotImplementedError("Override this!")


class LinearElastic(PureMechanicsInterface):
    def __init__(self, n, dim, E, nu):
        self.C = Something()

    def evaluate_ip(self, ip_strain, ip):
        return self.C @ ip_strain, self.C


class LocalDamage(PureMechanicsInterface):
    def __init__(self, n, dim, damage_law):
        self.C = Something()
        self.kappa = Something()

    def static_evaluate(self, eps, kappa):
        """
        Actual constitutive law on integration point level
        """
        new_kappa = Something()
        dsigma = (1 - self.damage_law(new_kappa)) * C
        sigma = dsigma @ eps
        return sigma, dsigma, new_kappa

    def evaluate_ip(self, ip_strain, ip):
        sigma, dsigma, self.kappa[ip] = self.static_evaluate(ip_strain, self.kappa[ip])
        return sigma, dsigma


class Plasticity(PureMechanicsInterface):
    def __init__(self, n, dim, yield_f):
        self.C = Something()
        self.eps_plastic = Something()
        self.lmbda = Something()
        pass

    def static_evaluate(self, eps, eps_plastic, lmbda):
        """
        Actual constitutive law on integration point level
        """
        # Returnmapping calculates sigma, dsigma ...
        sigma_tr = self.C @ (eps - eps_plastic)
        sigma = Something()
        dsigma = Something()
        # and new history data
        eps_plastic_new = Something()
        lmbda_new = Something()
        return sigma, dsigma, eps_plastic_new, lmbda_new

    def evaluate_ip(self, ip_strain, ip):
        sigma, dsigma, self.eps_plastic[ip], self.lmbda[ip] = self.static_evaluate(
            ip_strain, self.eps_plastic[ip], self.lmbda[ip]
        )
        return sigma, dsigma


######
## OR
######


class PlasticityIpInterface:
    def __init__(self, material_parameters, yield_f, flow_rule):
        pass

    def evaluate(self, eps, eps_plastic, lmbda):
        """
        Actual constitutive law on integration point level
        """
        # Returnmapping calculates sigma, dsigma ...
        sigma_tr = self.C @ (eps - eps_plastic)
        sigma = Something()
        dsigma = Something()
        # and new history data
        eps_plastic_new = Something()
        lmbda_new = Something()
        return sigma, dsigma, eps_plastic_new, lmbda_new


class Plasticity(PureMechanicsInterface):
    def __init__(self, n, dim, plasticity_ip):
        # allocate same history data for all plasticity models?
        self.eps_plastic = np.zeros((n, dim))
        self.lmbda = np.zeros(n)
        pass

    def evaluate_ip(self, ip_strain):
        (
            sigma,
            dsigma,
            self.eps_plastic[ip],
            self.lmbda[ip],
        ) = self.plasticity_ip.evaluate(ip_strain, self.eps_plastic[ip], self.lmbda[ip])

        return sigma, dsigma
