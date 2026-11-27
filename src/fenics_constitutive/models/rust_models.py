from __future__ import annotations

import numpy as np

from fenics_constitutive._bindings import (
    PyDruckerPrager3D,
    PyDruckerPragerHyperbolic3D,
    PyLinearElasticity3D,
    PyMisesPlasticity3D,
    PyIsotropicMises3D,
)

from .interfaces import IncrSmallStrainModel, StressStrainConstraint

__all__ = ["LinearElasticity3D", "MisesPlasticityLinearHardening3D"]


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
    r"""
    A classic Drucker-Prager plasticity model for 3D stress states.

    This class represents the Drucker-Prager yield criterion with associated flow rule.
    The yield function is defined as $f = \sqrt{J_2} + b \cdot I_1 - a$, where:

    - $J_2$ is the second invariant of the deviatoric stress tensor
    - $I_1$ is the first invariant of the stress tensor
    - $a$ and $b$ are material parameters that describe the yield surface.
    - $b_{flow}$ defines the slope of the flow rule which is equal to $b$ for associated flow.
      For $b_{flow} = 0$ the return direction is purely deviatoric (radial return algorithm)

    Yield surface: meridian section ($I_1$ vs. $\sqrt{J_2}$); the deviatoric section is a circle.
    The cone has the slope $-b$ and its tip at $I_1 = a/b$: $a$ shifts the cone, and $b$ rotates
    it around the point $(0, a)$. $b_{flow}$ only changes the flow direction.

    ![Yield surface of the Drucker-Prager model](../figures/yield_surface_drucker_prager.svg)

    Args:
        parameters (dict[str, np.ndarray]): A dictionary containing:

            - "mu": Shear modulus
            - "kappa": Bulk modulus
            - "a": Yield strength at zero pressure
            - "b": slope of the yield surface in the $I_1$-$\sqrt{J_2}$ plane
            - "b_flow": slope of the flow-potential, use $b_{flow} = b$ for associated flow
    """


@fenics_constitutive_wrapper(PyDruckerPragerHyperbolic3D)
class DruckerPragerHyperbolic3D(IncrSmallStrainModel):
    r"""
    A hyperbolically approximated Drucker-Prager plasticity model for 3D stress states.

    This class represents the Drucker-Prager yield criterion with either associated or
    non-associated flow rule. The yield function is defined as
    $f = \sqrt{J_2 + (bd)^2} + b \cdot I_1 - a$, where:

    - $J_2$ is the second invariant of the deviatoric stress tensor
    - $I_1$ is the first invariant of the stress tensor
    - $a$ and $b$ are material parameters that describe the yield surface as in the
      DruckerPrager3D model. $d$ is an additional smoothing parameter for the tip.
    - $h$ is a hardening parameter that scales $a$ and $d$ by $1 + h\alpha$.
    - $b_{flow} > 0$ defines the slope of the flow rule which is equal to $b$ for associated flow.

    Yield surface: meridian section ($I_1$ vs. $\sqrt{J_2}$); the deviatoric section is a circle.
    Away from the apex, the hyperbola approaches the cone of the DruckerPrager3D model; the
    smoothing $d$ rounds off the tip and moves the apex from $I_1 = a/b$ to
    $I_1 = (a/b - d)(1 + h\alpha)$. The hardening scales $a$ and $d$ by $1 + h\alpha$, which
    moves the apex while keeping the slope $b$. $b_{flow}$ only changes the flow direction.

    ![Yield surface of the hyperbolic Drucker-Prager model](../figures/yield_surface_drucker_prager_hyperbolic.svg)

    The return mapping may fail to converge for tensile trial stresses that return into the
    smoothed apex region while lying far outside of it: starting from the trial stress, the Newton
    iteration returns mainly deviatorically and does not find the region of width
    $\sqrt{J_2} \sim bd(1 + h\alpha)$ around the apex. This region becomes narrower as $d \to 0$,
    where the yield surface approaches the non-differentiable apex of the cone, so the return
    mapping may fail for small $d$, in particular with large strain increments.

    Args:
        parameters (dict[str, np.ndarray]): A dictionary containing:

            - "mu": Shear modulus
            - "kappa": Bulk modulus
            - "a": Yield strength at zero pressure
            - "b": slope of the yield surface in the $I_1$-$\sqrt{J_2}$ plane
            - "d": Smoothing parameter
            - "h": Hardening parameter
            - "b_flow": slope of the flow-potential (positive), use $b_{flow} = b$ for associated flow
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

    Yield surface: meridian section ($I_1$ vs. $\sqrt{J_2}$); the deviatoric section is a circle.
    The yield surface does not depend on $I_1$, and the yield stress $y_0 + h\alpha$ moves it
    parallel to the $I_1$ axis.

    ![Yield surface of the von Mises model](../figures/yield_surface_mises.svg)

    Args:
        parameters (dict[str, np.ndarray]): A dictionary containing:
            
            - "mu": Shear modulus
            - "kappa": Bulk modulus
            - "y_0": Initial yield stress
            - "h": Linear hardening modulus
    """

@fenics_constitutive_wrapper(PyIsotropicMises3D)
class IsotropicMises3D(IncrSmallStrainModel):
    r"""
    A von Mises plasticity model with linear hardening for 3D stress states.

    This class implements the von Mises yield criterion with linear isotropic hardening.
    The yield function is defined as $f = \sqrt{3/2 \cdot s:s} - \sigma_y$, where:

    - $s$ is the deviatoric stress tensor
    - $\sigma_y = y_0 + h \cdot \alpha$ is the current yield stress
    - $\alpha$ is the equivalent plastic strain

    Yield surface: meridian section ($I_1$ vs. $\sqrt{J_2}$); the deviatoric section is a circle.
    The yield surface does not depend on $I_1$, and the yield stress $y_0 + h\alpha$ moves it
    parallel to the $I_1$ axis.

    ![Yield surface of the von Mises model](../figures/yield_surface_mises.svg)

    Args:
        parameters (dict[str, np.ndarray]): A dictionary containing:

            - "mu": Shear modulus
            - "kappa": Bulk modulus
            - "y_0": Initial yield stress
            - "h": Linear hardening modulus
    """
