import numpy as np
import cofe
from time import perf_counter
_timings = []

# mesh=dfx.mesh.create_unit_cube(MPI.COMM_WORLD,25,25,25)
# V=dfx.fem.VectorFunctionSpace(mesh,("CG",1))
# rule = h.QuadratureRule()
# strain = h.QuadratureEvaluator()
# def as_mandel(T):
    # """
    # T: 
        # Symmetric 3x3 tensor
    # Returns:
        # Vector representation of T with factor sqrt(2) for shear components
    # """
    # factor = 2 ** 0.5
    # return ufl.as_vector(
        # [
            # T[0, 0],
            # T[1, 1],
            # T[2, 2],
            # factor * T[1, 2],
            # factor * T[0, 2],
            # factor * T[0, 1],
        # ]
    # )
# def eps_ufl(v):
    # return ufl.sym(ufl.grad(v))

# def sigma(v, C):
    # e = as_mandel(eps_ufl(v))
    # return ufl.dot(C,e)


class TTimer:
    def __init__(self, what=""):
        self.what = what

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, *args):
        ms = (perf_counter() - self.start) * 1000
        _timings.append((self.what, ms))


n=1000000
loops=10
cofe_timer=TTimer("cofe")
numpy_timer=TTimer("numpy")

law=cofe.EigenLinearElastic(20.,0.3,True,n)
law2=cofe.LinearElastic(20.,0.3,True,n)
# u=dfx.fem.Function(V)
# D = ufl.as_matrix(law.D.to_list())

# u.vector.array[:]=np.random.random(u.vector.array.size)

# eval_strain = h.QuadratureEvaluator(eps_ufl(u), mesh, rule)
# eval_stress = h.QuadratureEvaluator(sigma(u,D), mesh, rule)

# eps = eval_strain().flatten()
# sigma = np.zeros_like(eps)
# Ct = np.zeros(eps.size*6)
eps =np.random.random(n*6)
sigma=np.zeros(n*6)
Ct = np.zeros(n*36)
def python_loop(eps, sigma, tangents):
    C=law.C
    for i in range(n):
        sigma[i*6:(i+1)*6] = C@eps[i*6:(i+1)*6]
        tangents[i*36:(i+1)*36] = C.flatten()

with TTimer("cofe_eigen") as timer:
    for i in range(loops):
        law.evaluate(eps,sigma,Ct,1.)

with TTimer("cofe_xtensor") as timer:
    for i in range(loops):
        law2.evaluate(eps,sigma,Ct,1.)

with TTimer("python_loop") as timer:
    for i in range(loops):
        python_loop(eps,sigma,Ct)

with TTimer("numpy") as timer:
    for i in range(loops):
        C=law.C
        Ct[:] = np.tile(C.flatten(),n)
        sigma_view = sigma.reshape(-1,6)
        np.matmul(eps.reshape(-1,6),C,sigma_view) 
# with TTimer("fenicsx") as timer:
    # eval_stress(sigma)
print(_timings)

C=law.C
print(C is law.C)
