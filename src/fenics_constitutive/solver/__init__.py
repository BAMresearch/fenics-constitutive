"""
fenics-constitutive: Interfaces for solver of own constitutive models following models interface for dolfinx
"""

from __future__ import annotations

from ._solver import IncrSmallStrainProblem
from .utils import *

__all__ = ["IncrSmallStrainProblem"]
