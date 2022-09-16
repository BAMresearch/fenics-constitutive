FEniCS Constitutive
===================

https://github.com/BAMresearch/fenics-constitutive.git

We show examples on how to implement complex constitutive models, e.g. 
plasticity with return mapping in each Gauss point, using FEniCS quadrature 
function spaces.

Motivation
----------

[FEniCS](https://fenicsproject.org/) provides a powerful finite element (FE)
framework, including an abstract, high-level language (UFL) to formulate PDEs. 
These forms are automatically compiled and linked to the rest of the FEniCS 
library. This makes it both easy-to-program and fast-to-run. 

The UFL, however, is limited to expressions that can directly be written as 
mathematical functions of the solution fields. In certain cases of constitutive 
mechanics modeling, this is a limitation, as it cannot be used to 

- implement return mapping
    - classically a Newton-Raphson algorithm on each Gauss point
- use formulations based on eigenvalues, e.g. the Rankine norm

Additionally, there is this feeling of less control over the actual code, as it
is automatically generated and not designed to be human-readable.

Installation
------------

*FEniCS Constitutive* provides an interface between [FEniCS](https://fenicsproject.org/) and a C++ implementation of user-defined constitutive models. The interface can also act as a shortcut to the user material subroutines UMAT of the [ABAQUS](https://www.3ds.com/products-services/simulia/products/abaqus/) finite element program. Whether a single UMAT subroutine or a collection of the material subroutines must be precompiled into a static library, and further linked to the *FEniCS Constitutive* interface.

### Installation with *labtools*

*labtools* is a collection of tools and subroutines written in Fortran 90 for a user-friendly modular implementation of constitutive laws (for isotropic responses, single-crystals plasticity etc.), including a library of UMATs. A shortened version of *labtools* can be cloned from `shared_lib` branch of `<https://git.bam.de/chaftaog/labtools-fenics.git>`. For further details see /doc/User_documentation_labtools.docx. The compilation requires the Intel compiler `ifort`. Once cloned, execute the `make all` command in the /labtools-fenics folder.

In the second step clone the respective branch of *FEniCS Constitutive*

`git clone -b umat https://github.com/BAMresearch/fenics-constitutive.git`

The further installation requires [eigen3]( https://eigen.tuxfamily.org/), [NumPy](https://numpy.org/), [meshio](https://github.com/nschloe/meshio) and installation of helpers

`pip3 install git+https://github.com/BAMResearch/fenics_helpers`

Now configurate the /fenics-consitutive/CMakeLists.txt file. When setting `CMAKE_CXX_FLAGS` adjust the path to the static Intel libraries in the /lib/intel64 directory. Further, provide the location of the *labtools* library /labtools-fenics/lib/libumat.a within the `set_target_properties` command. Finally install the interface by starting

`pip install --user -e .`

from the /fenics-constitutive folder. To verify the installation, perform the tests in /fenics-consitutive/test and /fenics-consitutive/test/umat by `python3 test_*.py`.

The implementation of some constitutive laws within the interface is also given for demonstration purposes.

References
----------

The main idea comes from a [Comet-Fenics](https://comet-fenics.readthedocs.io/en/latest/demo/plasticity_mfront/plasticity_mfront.py.html#global-problem-and-newton-raphson-procedure)
example that defines the momentum balance equation where the stresses are not 
an expression of the strains, but a generic quadrature space function that is
*filled* manually. Similarly, the algorithmic tangent is provided manually 
on a quadrature space.

In each global Newton-Raphson step, they

1) project the strains `eps = sym(grad(u))` into a quadrature function 
       space that now contains Nx4 numbers, where N is the number of Gauss 
       points 
2) use [MFront](https://github.com/thelfer/MFrontGenericInterfaceSupport) 
       to calculate the stresses (Nx4) and the tangents (Nx16)
3) assign the stresses and the tangents to their quadrature spaces
4) assemble the system and solve.
5) optional: post-processing

As we want to keep full control (and not fall back to another code generation 
tool), we replace step 2) by 

- vectorized `numpy` code, or, if that is not possible
- a loop over all N Gauss points, or, if that is too slow
- a C++ function/class 
