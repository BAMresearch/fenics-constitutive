![Tests](https://github.com/BAMresearch/fenics-constitutive/actions/workflows/pytest.yml/badge.svg) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13364955.svg)](https://doi.org/10.5281/zenodo.13364955)

# fenics-constitutive

This project  provides a framework for using nonlinear constitutive material models in dolfinx with the goal of making it easy to implement new models and to use these models interchangeably in a simulation. This is made possible by prescribing a simple interface for the constitutive models and by providing a simple wrapper around the dolfinx `NonlinearProblem` that uses the constitutive model to compute the residual and the stiffness matrix. The project is currently focussed on small strain models with the goal of supporting large deformations through the use of objective stress rates and, therefore, also supporting a subset of the functionality provided by Abaqus UMATs or Ansys material models.

Although the project contains some constitutive models -- and will contain more in the foreseeable future -- we are currently not focussed on creating a comprehensive library of models. Instead, through the simplicity of the provided interface which only uses `numpy.ndarray` as a complex datatype, we want to enable users to write their own models in any language that can be linked to Python, while still being able to use their models in simulation scripts that other users have written using our interface.



## Installation using conda/mamba

Clone the repository and create a new environment from the `environment.yml` file:

```bash
git clone https://github.com/BAMresearch/fenics-constitutive.git
cd fenics-constitutive
mamba env create -f environment.yml
```
This automatically installs the required dependencies and the package itself.

Alternatively, if you have all dependencies listed in the `environment.yml` file installed, you can install the package using `pip` after cloning:

```bash
git clone https://github.com/BAMresearch/fenics-constitutive.git
cd fenics-constitutive
pip install -e .
```

## Usage

Since this project is based on FEniCSx, a basic knowledge of using FEniCSx is required. Similarly to any other FEniCSx project, you need to create a mesh, function spaces and boundary conditions. Defining the weak form is handled by the `IncrSmallStrainProblem` class. However, you may write your own Newton solver.


```python
import dolfinx as df
from fenics_constitutive import (
    IncrSmallStrainProblem, 
    IncrSmallStrainModel, 
    StressStrainConstraint, 
)
from fenics_constitutive.models import LinearElasticityModel

youngs_modulus = 42.0
poissons_ratio = 0.3

mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
V = df.fem.VectorFunctionSpace(mesh, ("CG", 1))
u = df.fem.Function(V)
law = LinearElasticityModel(
   {"E": youngs_modulus, "nu": poissons_ratio},
   StressStrainConstraint.FULL,
)

def left_boundary(x):
   return np.isclose(x[0], 0.0)

def right_boundary(x):
   return np.isclose(x[0], 1.0)

dofs_left = df.fem.locate_dofs_geometrical(V, left_boundary)
dofs_right = df.fem.locate_dofs_geometrical(V, right_boundary)
bc_left = df.fem.dirichletbc(np.array([0.0, 0.0, 0.0]), dofs_left, V)
bc_right = df.fem.dirichletbc(np.array([0.01, 0.0, 0.0]), dofs_right, V)

problem = IncrSmallStrainProblem(
   law,
   u,
   [bc_left, bc_right],
   1,
)

solver = NewtonSolver(MPI.COMM_WORLD, problem)
n, converged = solver.solve(u)
problem.update()

```

Currently the Python package contains the following models:

1. Linear elasticity for uniaxial stress, uniaxial strain, plane stress, plane strain, and full 3D stress and strain states.
2. Mises plasticity with isotropic nonlinear hardening.
3. Two viscoelasticity models: Standard linear solid model in both Maxwell representation and Kelvin-Voigt representation. 

## Citing

If you use this package in your research, please cite it using the following bibtex entry:

```bibtex
@software{fenics_constitutive2024,
author       = {Diercks, Philipp and
                Robens-Radermacher, Annika and
                Rosenbusch, Sjard Mathis and
                Unger, Jörg F. and
                Saif-Ur-Rehman},
title        = {fenics-constitutive},
month        = oct,
year         = 2024,
publisher    = {Zenodo},
doi          = {10.5281/zenodo.13364955},
url          = {https://doi.org/10.5281/zenodo.13364955},
}
```

If you want to cite a specific version, you can find the DOI on the Zenodo page.
