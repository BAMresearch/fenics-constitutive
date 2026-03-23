from __future__ import annotations

import dolfinx as df
import numpy as np
import ufl
from mpi4py import MPI


def norm(f: df.fem.Function, dx: ufl.Measure, comm: MPI.Comm = MPI.COMM_WORLD, type: str = "l2"):
    """
    Compute the norm of a function.

    Args:
    f: The function to compute the norm of.
    dx: The measure to integrate over.
    comm: The MPI communicator to use for parallel reduction. Default is `MPI.COMM_WORLD`.
    type: The type of norm to compute. Currently supports "l2" and "inf". Default is "l2".
    """
    match type:
        case "l2":
            norm_squared = df.fem.assemble_scalar(df.fem.form(ufl.inner(f, f) * dx))
            return np.sqrt(comm.allreduce(norm_squared, op=MPI.SUM))
        case "inf":
            norm_max = np.linalg.norm(f.x.array, ord=np.inf)
            return comm.allreduce(norm_max, op=MPI.MAX)
        case _:
            raise ValueError(f"Unknown norm type: {type}")
