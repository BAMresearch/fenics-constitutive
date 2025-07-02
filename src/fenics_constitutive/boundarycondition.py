from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import dolfinx as df
import numpy as np
import numpy.typing as npt
import ufl

BoundaryDescription = Callable[[npt.NDArray[np.number]], npt.NDArray[np.bool]]
TestFunction = ufl.Coargument | ufl.Argument


@dataclass(slots=True)
class NeumannBC:
    value: df.fem.Constant
    boundary: BoundaryDescription
    marker: int = 1

    def term(self, test_function: TestFunction, dA: ufl.Measure) -> ufl.Form:
        """
        Returns the Neumann boundary term for the given test function and measure.
        """
        return ufl.inner(self.value, test_function) * dA(self.marker)
