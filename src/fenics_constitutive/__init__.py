"""
Copyright (c) 2023 Sjard Mathis Rosenbusch. All rights reserved.

fenics-constitutive: Interfaces for constitutive models for dolfinx
"""


from __future__ import annotations

from .interfaces import *
from .stress_strain import *
from ._version import version as __version__

__all__ = ["__version__"]
