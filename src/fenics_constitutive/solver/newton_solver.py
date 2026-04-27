

from fenics_constitutive.solver import IncrSmallStrainGradientProblem
from petsc4py import PETSc
import dolfinx as df
from dolfinx.fem.petsc import NonlinearProblem, create_matrix, create_vector
from typing import Any, cast

class NewtonLinesearch:

    def __init__(self, comm, problem: NonlinearProblem, petsc_options=None):
                # Create matrix and vector to be used for assembly
        # of the non-linear problem
        self.problem = problem

        self._A = create_matrix(cast(Any, self.problem._a))
        
        self._b = create_vector(cast(Any, self.problem._L))

        # if u is None:
        #     # Extract function space from TrialFunction (which is at the
        #     # end of the argument list as it is numbered as 1, while the
        #     # Test function is numbered as 0)
        #     self.u = _Function(a.arguments()[-1].ufl_function_space())
        # else:
        #     self.u = u

        #self._x = la.create_petsc_vector_wrap(self.u.x)
        #self.bcs = bcs

        self._solver = PETSc.KSP().create(comm)  # type: ignore[attr-defined]
        self._solver.setOperators(self._A)

        # Give PETSc solver options a unique prefix
        problem_prefix = f"dolfinx_solve_{id(self)}"
        self._solver.setOptionsPrefix(problem_prefix)

        # Set PETSc options
        opts = PETSc.Options()  # type: ignore[attr-defined]
        opts.prefixPush(problem_prefix)
        if petsc_options is not None:
            for k, v in petsc_options.items():
                opts[k] = v
        opts.prefixPop()
        self._solver.setFromOptions()

        # Set matrix and vector PETSc options
        self._A.setOptionsPrefix(problem_prefix)
        self._A.setFromOptions()
        self._b.setOptionsPrefix(problem_prefix)
        self._b.setFromOptions()

    def __del__(self):
        self._solver.destroy()
        self._A.destroy()
        self._b.destroy()
        #self._x.destroy()
    
    def solve(
        self,
        u: df.fem.Function,
        max_it: int = 50,
        rtol: float = 1e-8,
        atol: float = 1e-10,
        max_ls_it: int = 12,
    ) -> tuple[int, bool]:
        """Solve the nonlinear problem with Newton's method and backtracking.

        Returns:
            A tuple ``(num_iterations, converged)``.
        """
        x_petsc = u.x.petsc_vec
        x0 = u.copy()
        x0_petsc = x0.x.petsc_vec

        dx = self._b.duplicate()
        b_trial = self._b.duplicate()

        converged = False
        r0_norm = 1.0

        for i in range(max_it):
            self.problem.form(x_petsc)
            self.problem.F(x_petsc, self._b)
            r_norm = self._b.norm()

            if i == 0:
                r0_norm = max(r_norm, 1.0)

            print("iteration", i, "r_norm", r_norm, "r_norm_rel",r_norm/r0_norm)

            if r_norm <= atol + rtol * r0_norm:
                converged = True
                u.x.scatter_forward()
                break

            self.problem.J(x_petsc, self._A)

            # Solve J * dx = -F.
            self._b.scale(-1.0)
            self._solver.solve(self._b, dx)

            x_petsc.copy(result=x0_petsc)

            alpha = 1.0
            accepted = False
            for _ in range(max_ls_it):
                x0_petsc.copy(result=x_petsc)
                x_petsc.axpy(alpha, dx)

                self.problem.form(x_petsc)
                self.problem.F(x_petsc, b_trial)
                r_trial_norm = b_trial.norm()

                if r_trial_norm < r_norm:
                    print("line search completed with alpha", alpha," r_norm", r_trial_norm, "r_norm_rel",r_trial_norm/r0_norm)
                    b_trial.copy(result=self._b)
                    accepted = True
                    break

                alpha *= 0.5
                print("line search started with r_norm", r_trial_norm, "r_norm_rel",r_trial_norm/r0_norm)

            if not accepted:
                x0_petsc.copy(result=x_petsc)
                break

            u.x.scatter_forward()

        dx.destroy()
        b_trial.destroy()

        return i + 1, converged

