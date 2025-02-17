
# Linear elasticity in Rust

All files for this example can be found in the [GitHub repository](https://github.com/BAMresearch/fenics-constitutive/tree/main/examples/elasticity_rs).

## Setting up the project

In order to build a linear elasticity model in Rust, you need to install the rust compiler which comes with the build tool Cargo. You can install it into your current conda-environment by running the following command:

```bash
mamba install rust -c conda-forge
```

After that you can create a new rust project using Cargo:

```bash
cargo new --lib elasticity_rs
```

This will create a new folder called `elasticity_rs` with the following structure:

```bash
elasticity_rs
├── Cargo.toml
└── src
    └── lib.rs
```

In order to write a constitutive model which is then linked to rust, you need to define the dependencies in the `Cargo.toml` file. For linear algebra on small matrices, the nalgebra crate is used. For interfacing with numpy, the numpy crate is used. The pyo3 crate is used to create a python module from the rust code. The dependencies are defined as follows:

```toml
--8<-- "examples/elasticity_rs/Cargo.toml"
```

Note that `crate-type = ["cdylib"]` is used to create a dynamic library which can be imported into python. 

## Writing the model

An example for a full source code of an elasticity model in Rust is shown below:

```rust
--8<-- "examples/elasticity_rs/src/lib.rs"
```


## Compilation

The rust code can be compiled using Cargo:

```bash
cargo build --release
```

This creates the shared library `target/release/libelasticity_rs.so`. This shared library cannot be imported directly, since the name of the shared library is not the same as the name of the module. To fix this, you can create a link to the shared library with the following command:

```bash
ln -f target/release/libelasticity_rs.so target/release/elasticity_rs.so
```

## Importing the shared library into Python

If the created shared library is located within the same directory as your Python script, you can import it directly with the following code:

```python
import elastictity_rs
```

You may also create a soft or hardlink to the shared library in your Python script directory, or import the script using the path to the shared library with 

```python
import importlib.util
from pathlib import Path
path_as_string = "..."
module_path = Path(path_as_string)
print(module_path.stem)
spec = importlib.util.spec_from_file_location(
    module_path.stem.split(".")[0], module_path
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

model = module.Elasticity3D(...)
```

## Create an IncrSmallStrainModel

So far, we have created a class that can compute the stress and tangent stiffness for a given strain, however, since it is a Rust extension, Python cannot recognise it as a `IncrSmallStrainModel`. This is important because the `IncrSmallStrainProblem` requires the method `constraint` to determine the modeling assumptions about the strains and stresses that the model is implemented in. To do this, we need to create a Python class that inherits from `IncrSmallStrainModel` and calls the Rust extension. The full source code of the model is shown below:

```python
from elastictity_rs import Elasticity3D
from fenics_constitutive import IncrSmallStrainModel, StressStrainConstraint

class PyElasticity3D(IncrSmallStrainModel):
    def __init__(self, E, nu):
        self._model = Elasticity3D(E, nu)
    
    def evaluate(
        self,
        t: float,
        del_t: float,
        grad_del_u: np.ndarray,
        stress: np.ndarray,
        tangent: np.ndarray,
        history: dict[str, np.ndarray] | None,
    ) -> None:
        self._model.evaluate(t, del_t, grad_del_u, stress, tangent, history)
     
    @property
    def history_dim(self):
        return self._model.history_dim()
    
    @property
    def constraint(self):
        return StressStrainConstraint.FULL
```