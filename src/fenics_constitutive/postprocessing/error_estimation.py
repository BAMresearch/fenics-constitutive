from __future__ import annotations

import dolfinx as df
import numpy as np
import ufl
from mpi4py import MPI

__all__ = ["norm"]


def norm(f, dx, comm=MPI.COMM_WORLD, norm_type="l2"):
    match norm_type:
        case "l2":
            norm_squared = df.fem.assemble_scalar(df.fem.form(ufl.inner(f, f) * dx))
            return np.sqrt(comm.allreduce(norm_squared, op=MPI.SUM))
        case "inf":
            norm_max = np.linalg.norm(f.x.array, ord=np.inf)
            return comm.allreduce(norm_max, op=MPI.MAX)
        case _:
            msg = f"Unknown norm type: {type}"
            raise ValueError(msg)
