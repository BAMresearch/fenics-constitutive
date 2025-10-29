from __future__ import annotations

import numpy as np

from fenics_constitutive._bindings import (
    PyDruckerPrager3D,
    PyDruckerPragerHyperbolic3D,
    PyLinearElasticity3D,
    PyMisesPlasticity3D,
)

from .interfaces import IncrSmallStrainModel, StressStrainConstraint

__all__ = ["LinearElasticity3D", "MisesPlasticity3D"]


def fenics_constitutive_wrapper(rust_model):
    def decorator(cls):
        assert issubclass(cls, IncrSmallStrainModel), (
            "decorator can only be used on subclasses of IncrSmallStrainModel"
        )

        # Overwrite __init__
        def __init__(self, parameters: np.ndarray) -> None:
            self.model = rust_model(parameters)
            self._constraint = StressStrainConstraint[
                str(self.model.constraint).split(".")[-1]
            ]
            assert (
                self._constraint.stress_strain_dim
                == self.model.constraint.stress_strain_dim
            )
            assert (
                self._constraint.geometric_dim == self.model.geometric_dim
            )


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
        assert (
            "evaluate" in cls.__abstractmethods__
        )
        assert (
            "constraint" in cls.__abstractmethods__
        )
        assert (
            "history_dim" in cls.__abstractmethods__
        )
        assert (
            len(cls.__abstractmethods__) == 3
        )

        # empty the abstract methods field to signal that all methods are overwritten
        cls.__abstractmethods__ = frozenset()
        return cls

    return decorator


@fenics_constitutive_wrapper(PyLinearElasticity3D)
class LinearElasticity3D(IncrSmallStrainModel):
    """
    A linear elasticity model written in Rust.

    Args:
       parameters (np.ndarray): [mu, kappa] an array containing the shear modulus and the bulk modulus.
    """


@fenics_constitutive_wrapper(PyDruckerPrager3D)
class DruckerPrager3D(IncrSmallStrainModel):
    pass

@fenics_constitutive_wrapper(PyDruckerPragerHyperbolic3D)
class DruckerPragerHyperbolic3D(IncrSmallStrainModel):
    pass


@fenics_constitutive_wrapper(PyMisesPlasticity3D)
class MisesPlasticity3D(IncrSmallStrainModel):
    pass
