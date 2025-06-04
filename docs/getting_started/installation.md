# Installation

The package is currently not available on `PyPI` or `conda-forge`, however, you can install it from source. At the moment the package is only tested on Linux, since it relies on `dolfinx` which is not available on Windows.
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