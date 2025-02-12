# fenics-constitutive

This project  provides a framework for using nonlinear constitutive material models in dolfinx with the goal of making it easy to implement new models and to use these models interchangably in a simulation. This is made possible by prescribing a simple interface for the constitutive models and by providing a simple wrapper around the dolfinx `NonlinearProblem` that uses the constitutive model to compute the residual and the stiffness matrix. The project is currently focussed on small strain models with the goal of supporting large deformations through the use of objective stress rates  and therefore also supporting a subset of the functionality provided by Abaqus UMATs or Ansys material models.

Although the project contains some constitutive models -- and will contain more in the foreseeable future -- we are currently not focussed on creating a comprehensive library of models. Instead, through the simplicity of the provided interface which only uses `numpy.ndarray` as a complex datatype, we want to enable users to write their own models in any language that can be linked to Python, while still being able to use their models in simulation scripts that other users have written using our interface.



<!--The main contribution is currently the `IncrSmallStrainModel` which is a simple interface for incrementally formulated (meaning $\Delta\varepsilon$ instead of $\varepsilon$) small strain constitutive models. It is formulated such that it can be easily used with objective stress rates and therefore large deformations as well. These models are then provided to the `IncrSmallStrainProblem` which is a simple wrapper around the dolfinx `NonlinearProblem` that uses the constitutive model to compute the residual and the tangent. This `IncrementalSmallStrainProblem` can then be used in the `dolfinx.nls.NewtonSolver`.-->

<!--In an earlier version which is still accesible on the old `master` branch, only constitutive models that were written in C++ and compiled together with the whole package could be used. This meant that there was no flexible way of extending the package with new constitutive models without contributing to this repository.-->

<!--The new version is designed with the goal of having minimal restrictions to writing new models. We achieve this by using Python as the main language for the interface and by using `numpy.ndarray` as the main data type for the constitutive models. Using this library with your own constitutive model is as simple as extending the `IncrSmallStrainModel` class and implementing the `evaluate` method. This approach allows for models to be written in `C++` (via `pybind11` or `nanobind`), `Fortran`, `Rust` (via `PyO3`) and many other languages as long as they can be linked to Python. -->

