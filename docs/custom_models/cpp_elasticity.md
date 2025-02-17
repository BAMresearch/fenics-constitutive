
# Linear elasticity in C++

All files for this example can be found in the [GitHub repository](https://github.com/BAMresearch/fenics-constitutive/tree/main/examples/elasticity_cpp).

## Setting up the project

In order to build a linear elasticity model in C++, you need to install a C++ compiler and the build tool Cmake. Additionally, you need a library for linear algebra, and a library to generate Python bindings. We recommend Eigen for linear algebra and either pybind11 or nanobind for the Python bindings. You can install it into your current conda-environment by running the following command:

```bash
mamba install cmake, clang, pybind11, eigen -c conda-forge
```

After that you can create a new C++ project with the following commands:

```bash
mkdir elasticity_cpp
mkdir elasticity_cpp/src
touch elasticity_cpp/CMakeLists.txt
touch src/main.cpp
```

This will create a new folder called `elasticity_cpp` with the following structure:

```bash
elasticity_cpp
├── CMakeLists.txt
└── src
    └── main.cpp
```

In order to write a constitutive model which is then linked to Python, you need to define the dependencies in the `CMakeLists.txt` file. For linear algebra on small matrices, the Eigen library is used. For interfacing with Python, the pybind11 library is used:

```cmake
--8<-- "examples/elasticity_cpp/CMakeLists.txt"
```


## Writing the model

An example for a full source code of an elasticity model in C++ is shown below:

```cpp linenums="1"
--8<-- "examples/elasticity_cpp/src/main.cpp"
```

## Compilation

To compile the C++ code, you need to create a build directory and run CMake:

```bash
cmake -DCMAKE_CXX_COMPILER=clang -S src/main.cpp -B build
cmake --build build
```

This creates the shared library `build/elasticity_cpp.platform_tag.so` where `platform_tag` is the platform specific tag of your system. 

## Importing the shared library into Python

If the created shared library is located within the same directory as your Python script, you can import it directly with the following code:

```python
import elastictity_cpp
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

So far, we have created a class that can compute the stress and tangent stiffness for a given strain, however, since it is a C++ extension, Python cannot recognise it as a `IncrSmallStrainModel`. This is important because the `IncrSmallStrainProblem` requires the method `constraint` to determine the modeling assumptions about the strains and stresses that the model is implemented in. To do this, we need to create a Python class that inherits from `IncrSmallStrainModel` and calls the C++ extension. The full source code of the model is shown below:

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