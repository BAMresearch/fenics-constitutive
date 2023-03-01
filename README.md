[![Not Maintained](https://img.shields.io/badge/Maintenance%20Level-Abandoned-orange.svg)](https://gist.github.com/cheerfulstoic/d107229326a01ff0f333a1d3476e068d)

**This repository is no longer maintained, some of its features may be added to another repository in the future. This will be linked at a later point**

FEniCS Constitutive
===================


https://bamresearch.github.io/fenics-constitutive

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
