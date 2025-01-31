[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13364955.svg)](https://doi.org/10.5281/zenodo.13364955)

# fenics-constitutive

This project  provides a framework for using nonlinear constitutive material models in dolfinx. 
The main contribution is currently the `IncrSmallStrainModel` which is a simple interface for incrementally formulated (meaning $\Delta\varepsilon$ instead of $\varepsilon$) small strain constitutive models. It is formulated such that it can be easily used with objective stress rates and therefore large deformations as well. These models are then provided to the `IncrSmallStrainProblem` which is a simple wrapper around the dolfinx `NonlinearProblem` that uses the constitutive model to compute the residual and the tangent. This `IncrementalSmallStrainProblem` can then be used in the `dolfinx.nls.NewtonSolver`.

In an earlier version which is still accesible on the old `master` branch, only constitutive models that were written in C++ and compiled together with the whole package could be used. This meant that there was no flexible way of extending the package with new constitutive models without contributing to this repository.

The new version is designed with the goal of having minimal restrictions to writing new models. We achieve this by using Python as the main language for the interface and by using `numpy.ndarray` as the main data type for the constitutive models. Using this library with your own constitutive model is as simple as extending the `IncrSmallStrainModel` class and implementing the `evaluate` method. This approach allows for models to be written in `C++` (via `pybind11` or `nanobind`), `Fortran`, `Rust` (via `PyO3`) and many other languages as long as they can be linked to Python. 


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
from fenics_constitutive import IncrSmallStrainProblem, IncrSmallStrainModel, Constraint, strain_from_grad_u

class LinearElasticityModel3D(IncrSmallStrainModel):
    def __init__(self, parameters: dict[str, float]):
      E = parameters["E"]
      nu = parameters["nu"]
      mu = E / (2.0 * (1.0 + nu))
      lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

      self.D = np.array(
         [
            [2.0 * mu + lam, lam, lam, 0.0, 0.0, 0.0],
            [lam, 2.0 * mu + lam, lam, 0.0, 0.0, 0.0],
            [lam, lam, 2.0 * mu + lam, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0 * mu, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0 * mu, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 2.0 * mu],
         ]
      )

    def evaluate(
        self,
        time: float,
        del_t: float,
        grad_del_u: np.ndarray,
        mandel_stress: np.ndarray,
        tangent: np.ndarray,
        history: np.ndarray | dict[str, np.ndarray] | None,
    ) -> None:
        assert (
            grad_del_u.size // (self.geometric_dim**2)
            == mandel_stress.size // self.stress_strain_dim
            == tangent.size // (self.stress_strain_dim**2)
        )
        n_gauss = grad_del_u.size // (self.geometric_dim**2)
        mandel_view = mandel_stress.reshape(-1, self.stress_strain_dim)
        strain_increment = strain_from_grad_u(grad_del_u, self.constraint)
        mandel_view += strain_increment.reshape(-1, self.stress_strain_dim) @ self.D
        tangent[:] = np.tile(self.D.flatten(), n_gauss)

    @property
    def constraint(self) -> Constraint:
        return Constraint.FULL

    @property
    def history_dim(self) -> None:
        return None

    def update(self) -> None:
        pass

mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
V = df.fem.VectorFunctionSpace(mesh, ("CG", 1))
u = df.fem.Function(V)
law = LinearElasticityModel3D(
   parameters={"E": youngs_modulus, "nu": poissons_ratio},
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

Currently the Python package itself does not contain any constitutive models, however, under `examples/` you can find implementations of Hooke's law, isotropic hardening plasticity and viscoplasticity. 

## Citing