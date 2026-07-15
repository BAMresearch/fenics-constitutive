"""
Adapter for using local `IncrSmallStrainModel`s with the
`dolfinx-external-operator <https://github.com/a-latyshev/dolfinx-external-operator>`_
package.

The stress becomes a `FEMExternalOperator` acting on the Mandel-strain increment
of the displacement relative to the last committed state. Following the pattern
of the dolfinx-external-operator demos, the model's combined return-mapping
(stress + tangent + history in one pass) is registered under the tangent
multi-index ``(1,)``: evaluating the Jacobian operators computes the tangent as
the operator value and pushes the stress into the residual coefficient as a
side effect, so the model is called exactly once per Newton iteration.

This module requires the optional dependency ``dolfinx-external-operator``.
"""

from __future__ import annotations

from collections.abc import Callable

import dolfinx as df
import numpy as np
import ufl
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc

try:
    from dolfinx_external_operator import (
        FEMExternalOperator,
        evaluate_external_operators,
        evaluate_operands,
        replace_external_operators,
    )
except ModuleNotFoundError as err:
    msg = (
        "fenics_constitutive.external_operator requires the optional dependency "
        "'dolfinx-external-operator', see environment.yml"
    )
    raise ModuleNotFoundError(msg) from err

from fenics_constitutive.models.interfaces import (
    IncrSmallStrainModel,
    StressStrainConstraint,
)
from fenics_constitutive.solver._spaces import ElementSpaces
from fenics_constitutive.solver.typesafe import fn_for
from fenics_constitutive.solver.utils import ufl_mandel_strain

__all__ = [
    "ExternalOperatorProblem",
    "IncrSmallStrainExternalOperator",
    "grad_del_u_from_mandel_strain",
]


def grad_del_u_from_mandel_strain(
    strain: np.ndarray, constraint: StressStrainConstraint
) -> np.ndarray:
    """
    Compute a displacement gradient whose symmetric part reproduces the given
    Mandel strain (the right inverse of `strain_from_grad_u` on symmetric
    gradients). Models only use the symmetric part of `grad_del_u`, so this
    loses no information.

    Args:
        strain: Flat array of Mandel strains for all quadrature points.
        constraint: Constraint that the model is implemented for.

    Returns:
        Flat array of displacement gradients for all quadrature points.
    """
    sdim = constraint.stress_strain_dim
    gdim = constraint.geometric_dim
    strain_view = strain.reshape(-1, sdim)
    grad_u = np.zeros((strain_view.shape[0], gdim * gdim))
    factor = 1.0 / 2**0.5
    match constraint:
        case (
            StressStrainConstraint.UNIAXIAL_STRAIN
            | StressStrainConstraint.UNIAXIAL_STRESS
        ):
            grad_u[:, 0] = strain_view[:, 0]
        case StressStrainConstraint.PLANE_STRAIN | StressStrainConstraint.PLANE_STRESS:
            grad_u[:, 0] = strain_view[:, 0]
            grad_u[:, 3] = strain_view[:, 1]
            grad_u[:, 1] = grad_u[:, 2] = factor * strain_view[:, 3]
        case StressStrainConstraint.FULL:
            grad_u[:, 0] = strain_view[:, 0]
            grad_u[:, 4] = strain_view[:, 1]
            grad_u[:, 8] = strain_view[:, 2]
            grad_u[:, 1] = grad_u[:, 3] = factor * strain_view[:, 3]
            grad_u[:, 2] = grad_u[:, 6] = factor * strain_view[:, 4]
            grad_u[:, 5] = grad_u[:, 7] = factor * strain_view[:, 5]
        case _:
            msg = f"Constraint {constraint} not supported"
            raise NotImplementedError(msg)
    return grad_u.reshape(-1)


class IncrSmallStrainExternalOperator:
    """
    Wraps an `IncrSmallStrainModel` as a `FEMExternalOperator` so it can be used
    in hand-written variational forms with the dolfinx-external-operator package.

    The wrapper owns the committed (previous increment) state: displacement
    `u0`, stress `stress_n` and the history variables. Each model evaluation
    starts from that committed state, so re-evaluating at different Newton
    iterates is safe. After a converged step, `update()` commits the current
    state, mirroring `IncrSmallStrainProblem.update()`.

    Args:
        model: The constitutive model.
        u: The displacement field, the unknown of the nonlinear problem.
        q_degree: The quadrature degree.
        del_t: The time increment between steps.
    """

    def __init__(
        self,
        model: IncrSmallStrainModel,
        u: df.fem.Function,
        q_degree: int,
        del_t: float = 1.0,
    ) -> None:
        self.model = model
        self.u = u
        self.u0 = u.copy()
        self.time = 0.0
        self.del_t = del_t

        mesh = u.function_space.mesh
        spaces = ElementSpaces.create(mesh, model.constraint, q_degree)
        self.stress_n = fn_for(spaces.stress_vector_space)
        self.metadata = {"quadrature_degree": q_degree, "quadrature_scheme": "default"}
        self.dxm = ufl.dx(metadata=self.metadata)

        self.strain_increment = ufl_mandel_strain(self.u - self.u0, model.constraint)
        self.stress_op = FEMExternalOperator(
            self.strain_increment,
            function_space=spaces.stress_vector_space,
            external_function=self._external_function,
        )

        sdim = model.constraint.stress_strain_dim
        self._n_points = self.stress_op.ref_coefficient.x.array.size // sdim
        self._history_n: dict[str, np.ndarray] | None = None
        self._history_trial: dict[str, np.ndarray] | None = None
        if model.history_dim is not None:
            self._history_n = {}
            for name, dim in model.history_dim.items():
                size = dim if isinstance(dim, int) else dim[0] * dim[1]
                self._history_n[name] = np.zeros(self._n_points * size)

        self._residual_operators: list[FEMExternalOperator] = []
        self._jacobian_operators: list[FEMExternalOperator] = []

    def create_forms(
        self,
        u_test: ufl.Argument,
        external_forces: list[ufl.Form] | None = None,
    ) -> tuple[ufl.Form, ufl.Form]:
        """
        Build the residual and Jacobian forms with all external operators
        replaced by their quadrature-space coefficients, ready for `df.fem.form`.

        Args:
            u_test: The test function of the displacement space.
            external_forces: Forms subtracted from the residual (body forces,
                Neumann boundary conditions).

        Returns:
            The replaced residual and Jacobian forms.
        """
        constraint = self.model.constraint
        residual = (
            ufl.inner(ufl_mandel_strain(u_test, constraint), self.stress_op) * self.dxm
        )
        if external_forces is not None:
            residual -= sum(external_forces)
        u_trial = ufl.TrialFunction(self.u.function_space)
        jacobian = ufl.algorithms.expand_derivatives(
            ufl.derivative(residual, self.u, u_trial)
        )
        residual_replaced, self._residual_operators = replace_external_operators(
            residual
        )
        jacobian_replaced, self._jacobian_operators = replace_external_operators(
            jacobian
        )
        return residual_replaced, jacobian_replaced

    def evaluate(self, evaluate_tangent: bool = True) -> None:
        """
        Evaluate the constitutive model at the current displacement. Call once
        per Newton iteration, before assembling the residual or Jacobian.

        Args:
            evaluate_tangent: If `True`, evaluate the Jacobian operators (which
                also updates the stress coefficient as a side effect). If
                `False`, only the stress is computed (`tangent=None` path).
        """
        operators = (
            self._jacobian_operators if evaluate_tangent else self._residual_operators
        )
        assert operators, "create_forms() must be called before evaluate()"
        evaluated_operands = evaluate_operands(operators)
        evaluate_external_operators(operators, evaluated_operands)

    def update(self) -> None:
        """
        Commit the current state (displacement, stress, history) to the
        previous state and advance the time. Call after each converged step.
        """
        self.u0.x.array[:] = self.u.x.array
        self.u0.x.scatter_forward()
        self.stress_n.x.array[:] = self.stress_op.ref_coefficient.x.array
        self.stress_n.x.scatter_forward()
        if self._history_n is not None:
            assert self._history_trial is not None, "evaluate() was never called"
            for name, values in self._history_n.items():
                values[:] = self._history_trial[name]
        self.time += self.del_t

    @property
    def stress_0(self) -> df.fem.Function:
        """The committed stress of the previous increment."""
        return self.stress_n

    @property
    def stress_1(self) -> df.fem.Function:
        """The stress of the current increment (the operator's coefficient)."""
        return self.stress_op.ref_coefficient

    @property
    def history(self) -> dict[str, np.ndarray] | None:
        """The committed history variables of the previous increment."""
        return self._history_n

    def _external_function(
        self, derivatives: tuple[int, ...]
    ) -> Callable[[np.ndarray], np.ndarray]:
        match derivatives:
            case (0,):
                return self._stress_impl
            case (1,):
                return self._tangent_impl
            case _:
                msg = f"No external function for derivatives {derivatives}"
                raise NotImplementedError(msg)

    def _evaluate_model(
        self, strain_increment: np.ndarray, tangent: np.ndarray | None
    ) -> np.ndarray:
        grad_del_u = grad_del_u_from_mandel_strain(
            strain_increment.reshape(-1), self.model.constraint
        )
        stress = self.stress_n.x.array.copy()
        history = (
            {name: values.copy() for name, values in self._history_n.items()}
            if self._history_n is not None
            else None
        )
        self.model.evaluate(self.time, self.del_t, grad_del_u, stress, tangent, history)
        self._history_trial = history
        return stress

    def _stress_impl(self, strain_increment: np.ndarray) -> np.ndarray:
        return self._evaluate_model(strain_increment, None)

    def _tangent_impl(self, strain_increment: np.ndarray) -> np.ndarray:
        sdim = self.model.constraint.stress_strain_dim
        tangent = np.zeros(self._n_points * sdim * sdim)
        stress = self._evaluate_model(strain_increment, tangent)
        np.copyto(self.stress_op.ref_coefficient.x.array, stress)
        self.stress_op.ref_coefficient.x.scatter_forward()
        return tangent


class ExternalOperatorProblem(NonlinearProblem):
    """
    Convenience `NonlinearProblem` driving an `IncrSmallStrainExternalOperator`
    with dolfinx's `NewtonSolver`, analogous to `IncrSmallStrainProblem`. For
    custom solvers, use the wrapper's `create_forms`/`evaluate`/`update`
    directly instead.

    Args:
        wrapper: The external-operator wrapper around the constitutive model.
        bcs: The Dirichlet boundary conditions.
        external_forces: Forms subtracted from the residual.
        form_compiler_options: The options for the form compiler.
        jit_options: The options for the JIT compiler.
    """

    def __init__(
        self,
        wrapper: IncrSmallStrainExternalOperator,
        bcs: list[df.fem.DirichletBC],
        external_forces: list[ufl.Form] | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
    ) -> None:
        self.wrapper = wrapper
        u_test = ufl.TestFunction(wrapper.u.function_space)
        residual, jacobian = wrapper.create_forms(u_test, external_forces)
        super().__init__(
            residual,
            wrapper.u,
            bcs=bcs,
            J=jacobian,
            form_compiler_options=form_compiler_options
            if form_compiler_options is not None
            else {},
            jit_options=jit_options if jit_options is not None else {},
        )

    def form(self, x: PETSc.Vec) -> None:
        """Update the displacement from the solution vector and evaluate the
        constitutive model. Called by the Newton solver each iteration."""
        super().form(x)
        if x is not self.wrapper.u.x.petsc_vec:
            x.copy(self.wrapper.u.x.petsc_vec)
        self.wrapper.u.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        self.wrapper.evaluate(evaluate_tangent=True)

    def update(self) -> None:
        """Commit the converged state, see `IncrSmallStrainExternalOperator.update`."""
        self.wrapper.update()
