"""
Copyright (c) 2023 Sjard Mathis Rosenbusch. All rights reserved.

fenics-constitutive: Interfaces for constitutive models for dolfinx
"""


from __future__ import annotations

from ._version import version as __version__
from .interfaces import *
from .maps import *
from .solver import *
from .stress_strain import *

__all__ = ["__version__"]
