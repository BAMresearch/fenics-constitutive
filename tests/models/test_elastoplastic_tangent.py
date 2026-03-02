
from __future__ import annotations

import dolfinx as df
import numpy as np
import pytest
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI

from fenics_constitutive.models import IncrSmallStrainModel
from fenics_constitutive.models.rust_models import (
    IsotropicMises3D,
    MisesPlasticityLinearHardening3D,
)
from fenics_constitutive.solver import IncrSmallStrainProblem
from fenics_constitutive.solver._lawonsubmesh import LawOnSubMesh, create_law_on_submesh


def create_history(law:IncrSmallStrainModel)-> dict[str,np.ndarray]:
    output ={}
    if law.history_dim is None:
        return output

    for key,value in law.history_dim.items():
        val = np.zeros(value).flatten()
        output[key]=val
    return output


def test_tangent_3d_uniaxial_strain():
    
    sigma_analytical = np.zeros(6)
    sigma_numerical = np.zeros(6)

    tangent_analytical = np.zeros(36)
    tangent_numerical = np.zeros(36)

    #parameters for steel
    matparam = {
        "kappa": 166.67,  # Bulk modulus
        "mu": 80.77,    # Shear modulus
        "y_0": 0.2,      # Initial yield stress
        "h": 10.0,       # Hardening modulus
    }
    matparam_arr = {
        "mu": np.array([matparam["mu"]]),
        "kappa": np.array([matparam["kappa"]]),
        "y_0":np.array([matparam["y_0"]]),
        "h":np.array([matparam["h"]]),
    }

    law_analytical = IsotropicMises3D(matparam_arr)
    law_numerical = MisesPlasticityLinearHardening3D(matparam_arr)

    # Create history variables
    history_analytical = create_history(law_analytical)
    history_numerical = create_history(law_numerical)

    # del_grad_u for uniaxial strain
    grad_del_u = np.zeros((3, 3))
    grad_del_u[0, 0] = 0.001  

    for i in range(10):
        law_analytical.evaluate(0.0,0.0, grad_del_u.flatten(),sigma_analytical,tangent_analytical, history_analytical)
        law_numerical.evaluate(0.0,0.0, grad_del_u.flatten(),sigma_numerical,tangent_numerical, history_numerical)
        # Compare stress
        assert np.allclose(sigma_analytical, sigma_numerical, atol=1e-12, rtol=1e-14), f"Stress mismatch at iteration {i}"
        # Compare tangent
        assert np.allclose(tangent_analytical, tangent_numerical, atol=1e-12, rtol=1e-14), f"Tangent mismatch at iteration {i}"

    assert history_analytical["history"][0] > 0, "plastic strain not reached in test"

    
def test_tangent_3d_plane_strain():
    
    sigma_analytical = np.zeros(6)
    sigma_numerical = np.zeros(6)

    tangent_analytical = np.zeros(36)
    tangent_numerical = np.zeros(36)

    #parameters for steel
    matparam = {
        "kappa": 166.67,  # Bulk modulus
        "mu": 80.77,    # Shear modulus
        "y_0": 0.2,      # Initial yield stress
        "h": 10.0,       # Hardening modulus
    }
    matparam_arr = {
        "mu": np.array([matparam["mu"]]),
        "kappa": np.array([matparam["kappa"]]),
        "y_0":np.array([matparam["y_0"]]),
        "h":np.array([matparam["h"]]),
    }

    law_analytical = IsotropicMises3D(matparam_arr)
    law_numerical = MisesPlasticityLinearHardening3D(matparam_arr)

    # Create history variables
    history_analytical = create_history(law_analytical)
    history_numerical = create_history(law_numerical)

    # del_grad_u for uniaxial strain
    grad_del_u = np.zeros((3, 3))
    grad_del_u[0, 0] = 0.001
    grad_del_u[1, 1] = 0.001

    for i in range(10):
        law_analytical.evaluate(0.0,0.0, grad_del_u.flatten(),sigma_analytical,tangent_analytical, history_analytical)
        law_numerical.evaluate(0.0,0.0, grad_del_u.flatten(),sigma_numerical,tangent_numerical, history_numerical)
        # Compare stress
        assert np.allclose(sigma_analytical, sigma_numerical, atol=1e-12, rtol=1e-14), f"Stress mismatch at iteration {i}"
        # Compare tangent
        assert np.allclose(tangent_analytical, tangent_numerical, atol=1e-12, rtol=1e-14), f"Tangent mismatch at iteration {i}"

    assert history_analytical["history"][0] > 0, "plastic strain not reached in test"


def test_tangent_3d_random():
    
    sigma_analytical = np.zeros(6)
    sigma_numerical = np.zeros(6)

    tangent_analytical = np.zeros(36)
    tangent_numerical = np.zeros(36)

    #parameters for steel
    matparam = {
        "kappa": 166.67,  # Bulk modulus
        "mu": 80.77,    # Shear modulus
        "y_0": 0.2,      # Initial yield stress
        "h": 10.0,       # Hardening modulus
    }
    matparam_arr = {
        "mu": np.array([matparam["mu"]]),
        "kappa": np.array([matparam["kappa"]]),
        "y_0":np.array([matparam["y_0"]]),
        "h":np.array([matparam["h"]]),
    }

    law_analytical = IsotropicMises3D(matparam_arr)
    law_numerical = MisesPlasticityLinearHardening3D(matparam_arr)

    # Create history variables
    history_analytical = create_history(law_analytical)
    history_numerical = create_history(law_numerical)

    # del_grad_u for uniaxial strain
    grad_del_u = np.random.random((3,3))
    grad_del_u = 0.001*grad_del_u/np.linalg.norm(grad_del_u)

    for i in range(10):
        law_analytical.evaluate(0.0,0.0, grad_del_u.flatten(),sigma_analytical,tangent_analytical, history_analytical)
        law_numerical.evaluate(0.0,0.0, grad_del_u.flatten(),sigma_numerical,tangent_numerical, history_numerical)
        # Compare stress
        assert np.allclose(sigma_analytical, sigma_numerical, atol=1e-12, rtol=1e-14), f"Stress mismatch at iteration {i}"
        # Compare tangent
        assert np.allclose(tangent_analytical, tangent_numerical, atol=1e-12, rtol=1e-14), f"Tangent mismatch at iteration {i}"

    assert history_analytical["history"][0] > 0, "plastic strain not reached in test"
