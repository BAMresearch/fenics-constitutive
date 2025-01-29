import ufl
import dolfinx as df
from mpi4py import MPI
import numpy as np

def norm(f, dx, comm=MPI.COMM_WORLD):
    norm_squared =  df.fem.assemble_scalar(df.fem.form(ufl.inner(f, f) * dx))
    return np.sqrt(comm.allreduce(norm_squared, op=MPI.SUM))
