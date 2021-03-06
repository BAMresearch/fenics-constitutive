"""
Note on iterative solvers
=========================

For highly nonlinear problems - like damage - configuring an iterative solver
(benefits: less memory, better scalability) is difficult. I am in no way
an expert in this, but I've found a setup that works.

It is based on using the (BiC)onjugate (G)radient (Stab)ilized method with
an (A)lgebraic (M)ulti(G)rid preconditioner. The latter one highly benefits
from setting a *near null space*, in our case the rigid body modes for linear
elasticity.

Note that this is just an adaptation of `this FEniCS example <https://bitbucket.org/fenics-project/dolfin/src/master/python/demo/undocumented/elasticity/demo_elasticity.py>`_ to 
the asymmetric and mixed gradient damage problem.
"""

from gradient_damage import *

def build_nullspace2D(V, u):
    """Function to build null space for 2D elasticity"""

    # Create list of vectors for null space
    nullspace_basis = [u.copy() for i in range(3)]
    # Build translational null space basis
    V.sub(0).dofmap().set(nullspace_basis[0], 1.0)
    V.sub(1).dofmap().set(nullspace_basis[1], 1.0)

    # Build rotational null space basis
    V.sub(0).set_x(nullspace_basis[2], -1.0, 1)
    V.sub(1).set_x(nullspace_basis[2], 1.0, 0)

    for x in nullspace_basis:
        x.apply("insert")

    # Create vector space basis and orthogonalize
    basis = VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()

    return basis

"""
We then modify the ``J`` method from the `original GDM <gradient_damage.html>`_
by subclassing ...
"""
class GDM_I(GDM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.null_space = build_nullspace2D(self.Vd, self.u.vector())

    def J(self, A, x):
        super().J(A, x)
        as_backend_type(A).set_near_nullspace(self.null_space)

"""
... and define an iterative solver.  
"""
if __name__ == "__main__":
    pc = PETScPreconditioner("petsc_amg")

    # Use Chebyshev smoothing for multigrid
    PETScOptions.set("mg_levels_ksp_type", "chebyshev")
    PETScOptions.set("mg_levels_pc_type", "jacobi")

    # Improve estimate of eigenvalues for Chebyshev smoothing
    PETScOptions.set("mg_levels_esteig_ksp_type", "gmres")
    PETScOptions.set("mg_levels_ksp_chebyshev_esteig_steps", 50)

    lin_solver = PETScKrylovSolver("bicgstab", pc)

    lin_solver.parameters["nonzero_initial_guess"] = True
    lin_solver.parameters["maximum_iterations"] = 1000
    lin_solver.parameters["relative_tolerance"] = 1.0e-6
    lin_solver.parameters["error_on_nonconvergence"] = False
    three_point_bending(GDM_I, lin_solver)

