from __future__ import annotations

import importlib.metadata

import fenics_constitutive as m


def test_version():
    assert importlib.metadata.version("fenics_constitutive") == m.__version__
