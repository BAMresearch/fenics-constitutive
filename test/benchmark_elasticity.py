from constitutiveX import cpp
import numpy as np
import time
import matplotlib.pyplot as plt

E=42.
nu=0.3
m = 20
sizes = [2**i for i in range(m)]
n_timings = 100

class TTimer:
    def __init__(self, what=""):
        self.what = what
        self.timings = []

    def evaluation(self):
        vec = np.array(self.timings)
        return {"mean": vec.mean(), "std": vec.std(), "measurements": vec.size}

    def total(self):
        return np.sum(self.timings)

    def mean(self):
        return np.mean(self.timings)

    def std(self):
        return np.std(self.timings)

    def to_array(self):
        return np.array(self.timings)

    def __str__(self):
        dic = self.evaluation()
        return f"Timer of {self.what},mean:{dic['mean']}\nstd:{dic['std']}\n#measurements:{dic['measurements']}"

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        # df.MPI.barrier(df.MPI.comm_world)
        ms = (time.perf_counter() - self.start) * 1000
        self.timings.append(ms)
timer_vf=TTimer("very fast model")
timer_map = TTimer("model with dictionaries")
timer_ref = TTimer("model with vectors")
for n in sizes:
    vf_law = cpp.VFLinearElastic3D(E,nu, n)
    map_law = cpp.MapLinearElastic3D({"E":E,"nu":nu},n)
    ref_law = cpp.LinearElastic3D(E,nu,n)
    eps = np.random.random(6*n)
    Ct = np.zeros(6*6*n)
    sigma_vf = np.zeros(6*n)
    sigma_map = sigma_vf.copy()
    sigma_ref = sigma_vf.copy()

    for j in range(n_timings):
        input = [np.array([])]*cpp.Q.LAST
        input[cpp.Q.EPS] = eps
        input[cpp.Q.SIGMA] = sigma_ref
        input[cpp.Q.DSIGMA_DEPS] = Ct
        input_dict = {"eps":eps,"sigma":sigma_map,"dsigma_deps":Ct}
        with timer_vf:
            vf_law.evaluate(eps,sigma_vf,Ct, 1.)
        with timer_map:
            map_law.evaluate(input_dict,input_dict, 1.)
        with timer_ref:
            ref_law.evaluate(input, 1.)
    # with timer_vf:
    #     for j in range(n_timings):
    #         vf_law.evaluate(eps,sigma_vf,Ct, 1.)
    # with timer_map:
    #     for j in range(n_timings):
    #         map_law.evaluate({"eps":eps,"sigma":sigma_map,"dsigma_deps":Ct},{}, 1.)
    # with timer_map:
    #     input = [np.array([])]*cpp.Q.LAST
    #     input[cpp.Q.EPS] = eps
    #     input[cpp.Q.SIGMA] = sigma_ref
    #     input[cpp.Q.DSIGMA_DEPS] = Ct
    #     for j in range(n_timings):
    #         ref_law.evaluate(input, 1.)
    del sigma_map, sigma_ref, sigma_vf, Ct, eps

mean_vf = np.mean(np.array(timer_vf.timings).reshape(-1,n_timings), axis=1)
mean_map = np.mean(np.array(timer_map.timings).reshape(-1,n_timings), axis=1)
mean_ref = np.mean(np.array(timer_ref.timings).reshape(-1,n_timings), axis=1)
plt.plot(sizes, mean_vf, label="very fast")
plt.plot(sizes, mean_map, label="with dictionaries")
plt.plot(sizes, mean_ref, label="with lists")
plt.legend()
plt.show()