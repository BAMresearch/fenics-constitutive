from __future__ import annotations

import dolfinx as df


def fn_for(space: df.fem.FunctionSpace) -> df.fem.Function:
    """Create a Function for the given FunctionSpace."""
    function = df.fem.Function(space)
    assert isinstance(function, df.fem.Function)
    return function
