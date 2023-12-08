# fenics-constitutive

_For some examples which are using legacy-FEniCS you may look on the old `master` branch [legacy code](https://github.com/BAMresearch/fenics-constitutive/tree/master)_

The new and improved version of `fenics-constitutive` which has the goal of simulating mechanical problems in FEniCSx with nonlinear consitutive models like plasticity, damage, etc. Everything is still a work in process, but from this projct you might expect:

1. An interface for constitutive models for small strain increments (and some examples for such models)
2. Solvers for mechanical problems which can use any constitutive model that follows our interface design
   * This means that **YOU** may write constitutive models in any programming language as long as you can bind that code to Python and this code takes `numpy.ndarray` as parameters.
3. Solvers that account for large deformations via an objective stress rate.
4. A documentaion explaining the interface design and tutorials on how to write your own constitutive models.

More to follow soon!
