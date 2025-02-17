# Linear elasticity as an Abaqus UMAT

All files for this example can be found in the [GitHub repository](https://github.com/BAMresearch/fenics-constitutive/tree/main/examples/umat).

## Setting up the project

In the preceding examples, we have implemented a linear elasticity model using one language and looped over all quadrature points. However, an Abaqus UMAT is only written for one quadrature point. Therefore, we need to write code implementing a for loop in which the UMAT is called for each quadrature point. This is done in a C++ class that is then compiled into a shared library. The shared library is then imported into Python as in the previous examples. 

In addition to the dependencies from the C++ example, we need to install a Fortran compiler. You can install it into your current conda-environment by running the following command:

```bash
mamba install gfortran -c conda-forge
```


After that you can create a new C++ project with the following commands:

```bash
mkdir umat
mkdir umat/src
touch umat/CMakeLists.txt
touch src/main.cpp
touch src/umat_linear_elastic.f
```

This will create a new folder called `umat` with the following structure:

```bash
umat
├── CMakeLists.txt
└── src
    ├── umat_linear_elastic.f
    └── main.cpp
```

In order to write a constitutive model with is then linked to Python, you need to define the dependencies in the `CMakeLists.txt` file. For linear algebra on small matrices, the `Eigen` library is used. For interfacing with Python, the `pybind11` library is used:

```cmake
--8<-- "examples/umat/CMakeLists.txt"
```


## Writing the model


An example for a full source code of an elasticity model as an Abaqus UMAT is shown below:

```fortran linenums="1"
--8<-- "examples/umat/src/umat_linear_elastic.f"
```

And the C++ code that calls the UMAT is shown below:

```cpp linenums="1"
--8<-- "examples/umat/src/main.cpp"
```

## Compilation

To compile the C++ code, you need to create a build directory and run CMake:

```bash
cmake -DCMAKE_CXX_COMPILER=clang -S src/main.cpp -B build
cmake --build build
```

This creates the shared library `build/umat.platform_tag.so` where `platform_tag` is the platform specific tag of your system. 

The Fortran code can be compiled using the following command:

```bash
gfortran -shared -fPIC -o build/umat_linear_elastic.so src/umat_linear_elastic.f
```

## Importing the shared library into Python

If the created shared library is located within the same directory as your Python script, you can import it directly with the following code:

```python
import umat
```

You may also create a soft or hard link to the shared library in your Python script directory, or import the script using the path to the shared library with 

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

So far, we have created a class that can compute the stress and tangent stiffness for a given strain, however, since it is a C++ extension, Python cannot recognize it as a `IncrSmallStrainModel`. This is important because the `IncrSmallStrainProblem` requires the method `constraint` to determine the modeling assumptions about the strains and stresses that the model is implemented in. To do this, we need to create a Python class that inherits from `IncrSmallStrainModel` and calls the C++ extension. The full source code of the model is shown below:

```python
from elastictity_cpp import Elasticity3D
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