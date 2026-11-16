from __future__ import annotations

import numpy as np

from fenics_constitutive._bindings import (
    PyDruckerPrager3D,
    PyDruckerPragerHyperbolic3D,
    PyHenckyDruckerPrager3D,
    PyHenckyDruckerPragerCauchyYield3D,
    PyHenckyDruckerPragerHyperbolic3D,
    PyHenckyDruckerPragerHyperbolicCauchyYield3D,
    PyHenckyElasticity3D,
    PyHenckyMises3D,
    PyHenckyMisesCauchyYield3D,
    PyLinearElasticity3D,
    PyMisesPlasticity3D,
    PyIsotropicMises3D,
)

from .interfaces import IncrSmallStrainModel, LargeStrainModel, StressStrainConstraint

__all__ = [
    "HenckyDruckerPrager3D",
    "HenckyDruckerPragerHyperbolic3D",
    "HenckyElasticity3D",
    "HenckyMises3D",
    "LinearElasticity3D",
    "MisesPlasticityLinearHardening3D",
]


def fenics_constitutive_wrapper(rust_model):
    def decorator(cls):
        assert issubclass(cls, IncrSmallStrainModel), (
            "decorator can only be used on subclasses of IncrSmallStrainModel"
        )

        # Overwrite __init__
        def __init__(self, parameters: dict[str, np.ndarray]) -> None:
            self.model = rust_model(parameters)
            self._constraint = StressStrainConstraint[
                str(self.model.constraint).split(".")[-1]
            ]
            assert (
                self._constraint.stress_strain_dim
                == self.model.constraint.stress_strain_dim
            )
            assert self._constraint.geometric_dim == self.model.geometric_dim

        # Add evaluate method
        def evaluate(
            self,
            t: float,
            del_t: float,
            grad_del_u: np.ndarray,
            stress: np.ndarray,
            tangent: np.ndarray,
            history: dict[str, np.ndarray] | None,
        ) -> None:
            self.model.evaluate(
                t,
                del_t,
                grad_del_u,
                stress,
                tangent,
                history,
            )

        # Add constraint property
        def constraint(self) -> StressStrainConstraint:
            return self._constraint

        # Add history_dim property
        def history_dim(self) -> dict[str, int | tuple[int, int]] | None:
            # Your implementation here
            return self.model.history_dim

        cls.__init__ = __init__

        cls.evaluate = evaluate

        cls.constraint = property(constraint)

        cls.history_dim = property(history_dim)

        # check that the only abstract fields in cls are the ones that we define
        assert "evaluate" in cls.__abstractmethods__
        assert "constraint" in cls.__abstractmethods__
        assert "history_dim" in cls.__abstractmethods__
        assert len(cls.__abstractmethods__) == 3

        # empty the abstract methods field to signal that all methods are overwritten
        cls.__abstractmethods__ = frozenset()
        return cls

    return decorator


@fenics_constitutive_wrapper(PyLinearElasticity3D)
class LinearElasticity3D(IncrSmallStrainModel):
    """
    A linear elasticity model written in Rust.

    Args:
       parameters (dict[str, np.ndarray]): A dictionary containing:
           - "mu": Shear modulus
           - "kappa": Bulk modulus
    """


@fenics_constitutive_wrapper(PyDruckerPrager3D)
class DruckerPrager3D(IncrSmallStrainModel):
    """
    A classic Drucker-Prager plasticity model for 3D stress states.

    This class represents the Drucker-Prager yield criterion with associated flow rule.
    The yield function is defined as: f = sqrt(J_2) + b·I_1 - a, where:
    - J_2 is the second invariant of the deviatoric stress tensor
    - I_1 is the first invariant of the stress tensor
    - a and b are material parameters that describe the yield surface.
    - b_flow defines the slope of the flow rule which is equal to b for associated flow.
      For b=0 the return direction is purely deviatoric (radial return algorithm)

    Args:
        parameters (dict[str, np.ndarray]): A dictionary containing:
            - "mu": Shear modulus
            - "kappa": Bulk modulus
            - "a": slope of the yield surface in I_1,sqrt(J_2) space
            - "b": Yield strength at zero pressure
            - "b_flow": slope of the flow-potential, use b_flow=b for associated flow
    """


@fenics_constitutive_wrapper(PyDruckerPragerHyperbolic3D)
class DruckerPragerHyperbolic3D(IncrSmallStrainModel):
    """
    A hyperbolically approximated Drucker-Prager plasticity model for 3D stress states.

    This class represents the Drucker-Prager yield criterion with either associated or non-associated flow rule.
    The yield function is defined as: f = sqrt(J_2+(bd)^2) + b·I_1 - a, where:
    - J_2 is the second invariant of the deviatoric stress tensor
    - I_1 is the first invariant of the stress tensor
    - a and b are material parameters that describe the yield surface as in the DruckerPrager3D model.
      d is an additional smoothing parameter for the tip.
    - b_flow defines the slope of the flow rule which is equal to b for associated flow.
      For b=0 the return direction is purely deviatoric (radial return algorithm)

    Args:
        parameters (dict[str, np.ndarray]): A dictionary containing:
            - "mu": Shear modulus
            - "kappa": Bulk modulus
            - "a": slope of the yield surface in I_1,sqrt(J_2) space
            - "b": Yield strength at zero pressure
            - "d": Smoothing parameter
            - "b_flow": slope of the flow-potential, use b_flow=b for associated flow
    """


@fenics_constitutive_wrapper(PyMisesPlasticity3D)
class MisesPlasticityLinearHardening3D(IncrSmallStrainModel):
    r"""
    A von Mises plasticity model with linear hardening for 3D stress states.

    This class implements the von Mises yield criterion with linear isotropic hardening.
    The yield function is defined as: $f = \sqrt{3/2 \cdot s:s} - \sigma_y$, where:
     
    - $s$ is the deviatoric stress tensor
    - $\sigma_y = y_0 + h \cdot \alpha$ is the current yield stress
    - $\alpha$ is the equivalent plastic strain

    Args:
        parameters (dict[str, np.ndarray]): A dictionary containing:
            
            - "mu": Shear modulus
            - "kappa": Bulk modulus
            - "y_0": Initial yield stress
            - "h": Linear hardening modulus
    """

@fenics_constitutive_wrapper(PyIsotropicMises3D)
class IsotropicMises3D(IncrSmallStrainModel):
    """
    A von Mises plasticity model with linear hardening for 3D stress states.

    This class implements the von Mises yield criterion with linear isotropic hardening.
    The yield function is defined as: f = sqrt(3/2 * s:s) - sigma_y, where:
    - s is the deviatoric stress tensor
    - sigma_y = y_0 + h * alpha is the current yield stress
    - alpha is the equivalent plastic strain

    Args:
        parameters (dict[str, np.ndarray]): A dictionary containing:
            - "mu": Shear modulus
            - "kappa": Bulk modulus
            - "y_0": Initial yield stress
            - "h": Linear hardening modulus
    """


def finite_strain_wrapper(rust_model, rust_model_cauchy_yield=None):
    """Adapts a compiled large-strain model to the `LargeStrainModel` interface. If
    `rust_model_cauchy_yield` is given, the class accepts `yield_stress="kirchhoff"` (default,
    `rust_model`) or `yield_stress="cauchy"` (`rust_model_cauchy_yield`)."""

    def decorator(cls):
        assert issubclass(cls, LargeStrainModel), (
            "decorator can only be used on subclasses of LargeStrainModel"
        )

        def __init__(
            self, parameters: dict[str, np.ndarray], yield_stress: str = "kirchhoff"
        ) -> None:
            models = {"kirchhoff": rust_model}
            if rust_model_cauchy_yield is not None:
                models["cauchy"] = rust_model_cauchy_yield
            if yield_stress not in models:
                msg = f"yield_stress must be one of {list(models)}, got {yield_stress!r}"
                raise ValueError(msg)
            self.yield_stress = yield_stress
            self.model = models[yield_stress](parameters)

        def evaluate(
            self,
            t: float,
            del_t: float,
            F: np.ndarray,
            tau: np.ndarray,
            tangent: np.ndarray | None,
            history: dict[str, np.ndarray] | None,
        ) -> None:
            self.model.evaluate(t, del_t, F, tau, tangent, history)

        def history_dim(self) -> dict[str, int | tuple[int, int]] | None:
            return self.model.history_dim

        def initial_history(self) -> dict[str, np.ndarray] | None:
            history = self.model.initial_history()
            if history is None:
                return None
            return {key: np.array(value) for key, value in history.items()}

        cls.__init__ = __init__
        cls.evaluate = evaluate
        cls.history_dim = property(history_dim)
        cls.initial_history = initial_history

        assert cls.__abstractmethods__ == frozenset(
            {"evaluate", "history_dim", "initial_history"}
        )
        cls.__abstractmethods__ = frozenset()
        return cls

    return decorator


@finite_strain_wrapper(PyHenckyElasticity3D)
class HenckyElasticity3D(LargeStrainModel):
    r"""
    Hencky hyperelasticity $\boldsymbol\tau = \kappa\ln J\,\mathbf I + 2\mu\,\mathrm{dev}\ln\mathbf V$.

    Args:
        parameters (dict[str, np.ndarray]): A dictionary containing:
            - "mu": Shear modulus
            - "kappa": Bulk modulus
    """


_YIELD_STRESS_DOC = """
        yield_stress (str): The stress at which yield function, flow direction and hardening
            function are evaluated: "kirchhoff" (default) for the Kirchhoff stress tau, "cauchy"
            for the Cauchy stress sigma = tau / det F. The elastic law is tau = C : eps_e in both
            cases. The results differ for volume changes, e.g. for pressure-dependent models
            with dilatancy."""


@finite_strain_wrapper(PyHenckyMises3D, PyHenckyMisesCauchyYield3D)
class HenckyMises3D(LargeStrainModel):
    __doc__ = (
        """
    Large-strain version of `IsotropicMises3D` (Hencky elasticity, multiplicative split,
    exponential-map return mapping).

    Args:
        parameters (dict[str, np.ndarray]): Same parameters as `IsotropicMises3D`."""
        + _YIELD_STRESS_DOC
    )


@finite_strain_wrapper(PyHenckyDruckerPrager3D, PyHenckyDruckerPragerCauchyYield3D)
class HenckyDruckerPrager3D(LargeStrainModel):
    __doc__ = (
        """
    Large-strain version of `DruckerPrager3D` (Hencky elasticity, multiplicative split,
    exponential-map return mapping).

    Args:
        parameters (dict[str, np.ndarray]): Same parameters as `DruckerPrager3D`."""
        + _YIELD_STRESS_DOC
    )


@finite_strain_wrapper(
    PyHenckyDruckerPragerHyperbolic3D, PyHenckyDruckerPragerHyperbolicCauchyYield3D
)
class HenckyDruckerPragerHyperbolic3D(LargeStrainModel):
    __doc__ = (
        """
    Large-strain version of `DruckerPragerHyperbolic3D` (Hencky elasticity, multiplicative
    split, exponential-map return mapping).

    Args:
        parameters (dict[str, np.ndarray]): Same parameters as `DruckerPragerHyperbolic3D`."""
        + _YIELD_STRESS_DOC
    )
