"""
truss under its own dead weight

solve
    EA u''(x) = - ρgA on Ω

analytical solution
    u(x) = ρgl (x - x ** 2 / 2 / l) / E
    with Ω = (0, l)
"""
import dolfin as df
import numpy as np
import pytest

# MODULE "SENSOR"


class DisplacementFieldSensor:
    def measure(self, u, R):
        return u


class DisplacementSensor:
    def __init__(self, where):
        self.where = where

    def measure(self, u, R):
        return u(self.where)


class ForceSensor:
    def __init__(self, bc):
        self.dofs = list(bc.get_boundary_values().keys())

    def measure(self, u, R):
        load_local = np.sum(R[self.dofs])
        return load_local


# MODULE "EXPERIMENT"


class Experiment:
    def __init__(self):
        self.data = {}

    def add_sensor_data(self, sensor, data, ts=[1.]):
        self.data[sensor] = (data, ts)


class UniaxialTrussExperiment(Experiment):
    def __init__(self, parameters):
        super().__init__()
        self.mesh = df.IntervalMesh(1, 0.0, parameters["L"])
        self.bcs = None  # to be created externally via .create_bcs

    def create_bcs(self, V):
        def left(x, on_boundary):
            return x[0] < df.DOLFIN_EPS and on_boundary
        
        def right(x, on_boundary):
            return x[parameters["L"]] > df.DOLFIN_EPS and on_boundary

        self.bcs = [df.DirichletBC(V, df.Constant(0.0), left)]
        return self.bcs

    def refine(self, N=1):
        """
        Refines the mesh `N` times.
        """
        for _ in range(N):
            self.mesh = df.refine(self.mesh)


def get_experiment(name, parameters):
    # metaprogramming!
    cls_name = name + "Experiment"
    return eval(cls_name)(parameters)


# MODULE "PROBLEM"


class LinearElasticity:
    def __init__(self, experiment, params):
        self.experiment = experiment
        self.params = params

        mesh = self.experiment.mesh
        self.V = df.FunctionSpace(mesh, "Lagrange", self.params["degree"])
        self.u = df.Function(self.V)
        v = df.TestFunction(self.V)
        self.bcs = self.experiment.create_bcs(self.V)

        rho, g, L, E, A = [
            df.Constant(self.params[what]) for what in ["rho", "g", "L", "E", "A"]
        ]
        F = params["rho"] * params["g"] * params["A"]
        self.f = df.Expression("t * F", t=0.0, F=F, degree=0)
        self.R = E * df.inner(df.grad(self.u), df.grad(v)) * df.dx - self.f * v * df.dx

    def solve(self, t=1.):
        self.f.t = t
        df.solve(
            self.R == 0,
            self.u,
            self.bcs,
            solver_parameters={"newton_solver": {"relative_tolerance": 1.0e-1}},
        )  # why??

        return self.u, df.assemble(self.R)

    def evaluate(self, sensors, t):
        """
        Evaluates the problem for the given sensors
        """
        u, R = self.solve(t)
        try:
            # only one sensor
            return sensors.measure(u, R)
        except AttributeError:
            # list of sensors
            return {s: s.measure(u, R) for s in sensors}

    def __call__(self, sensors, ts=[1.]):
        measurements = []
        for t in ts:
            measurements.append(self.evaluate(sensors, t))
        if len(ts) == 1:
            measurements = measurements[0]
        return measurements


class DisplacementSolution(df.UserExpression):
    def __init__(self, parameters):
        super().__init__(degree=8)
        self.parameters = parameters

    def eval(self, value, x):
        p = self.parameters
        rho, g, L, E, A = p["rho"], p["g"], p["L"], p["E"], p["A"]
        value[0] = rho * g * L * (x[0] - x[0] ** 2 / 2 / L) / E * A

    def value_shape(self):
        return ()


# EXAMPLE APPLICATION: "CONVERGENCE TEST"


def run_convergence(experiment, parameters, sensor, max_n_refinements=15, eps=1.0e-4):

    for n_refinements in range(max_n_refinements):
        problem = LinearElasticity(experiment, parameters)
        u_correct, ts = experiment.data[sensor]
        u_fem = problem(sensor, ts)

        try:
            # numpy ?
            err = np.linalg.norm(u_fem - u_correct)
        except TypeError:
            err = df.errornorm(u_correct, u_fem, norm_type="l2", mesh=experiment.mesh)

        if err < eps:
            break

        experiment.refine()
        n_refinements += 1

    print(f"Finally converged. Please use {n_refinements=}.")
    return n_refinements


# EXAMPLE APPLICATION: "PARAMETER ESTIMATION"

def estimate(experiment, parameters, sensor, what):
    from scipy.optimize import least_squares
    param = parameters.copy()

    def error(prm):
        param[what] = prm.squeeze()
        print(f"Try {what} = {param[what]}")
        problem = LinearElasticity(experiment, param)
        value_exp, ts = experiment.data[sensor]
        value_fem = problem(sensor, ts)
        return value_fem - value_exp

    optimize_result = least_squares(
        fun=error, x0=0.5 * param[what]
    )
    return optimize_result.x.squeeze()


def estimate_E(experiment, parameters, sensor):
    return estimate(experiment, parameters, sensor, what="E")


def test_fit_to_LD():
    parameters = {
        "L": 42.0,
        "E": 10.0,
        "g": 9.81,
        "A": 4.0,
        "rho": 7.0,
        "degree": 1,
    }
    experiment = get_experiment("UniaxialTruss", parameters)
    p = LinearElasticity(experiment, parameters)
    force_sensor = ForceSensor(p.bcs[0])

    F = p(force_sensor, ts=[1.])
    F_at_t_1 = -parameters["L"] * parameters["A"] * parameters["rho"] * parameters["g"]
    assert F == pytest.approx(F_at_t_1)

    ts = np.linspace(0, 1, 11)
    F_correct = F_at_t_1 * ts
    experiment.add_sensor_data(force_sensor, F_correct, ts)

    A = estimate(experiment, parameters, force_sensor, what="A")
    assert A == pytest.approx(parameters["A"])





def demonstrate_examples():
    parameters = {
        "L": 42.0,
        "E": 10.0,
        "g": 9.81,
        "A": 4.0,
        "rho": 7.0,
        "degree": 1,
    }
    experiment = get_experiment("UniaxialTruss", parameters)

    # attach analytic solution for the full displacement field
    full_u_sensor = DisplacementFieldSensor()
    u_correct = DisplacementSolution(parameters)
    experiment.add_sensor_data(full_u_sensor, u_correct)

    # attach analytic solution for the bottom displacement
    u_sensor = DisplacementSensor(where=parameters["L"])
    p = parameters
    u_max = 0.5 * p["rho"] * p["g"] * p["A"] * p["L"] ** 2 / p["E"]
    experiment.add_sensor_data(u_sensor, u_max)

    # First assume that we don't know E...
    parameters["E"] = 1.0
    # ...so we infer it using the measuremnts.
    parameters["E"] = estimate_E(experiment, parameters, u_sensor)

    # Run the convergence analysis with the maximum displacement. As for all
    # nodal values, we expect them to be correct even for a single linear
    # element.
    for degree in [1, 2, 3]:
        parameters["degree"] = degree
        n_refinements = run_convergence(experiment, parameters, u_sensor)
        assert n_refinements == 0

    # Run the convergence analysis with the whole displacement field.
    # Here, a linear solution can only be exact up to a given epsilon.
    # Quadratic and cubic interpolation caputure the field without refinement.
    for degree, expected_n_refinements in [(3, 0), (2, 0), (1, 8)]:
        parameters["degree"] = degree
        n_refinements = run_convergence(experiment, parameters, full_u_sensor, eps=1)
        assert n_refinements == expected_n_refinements


if __name__ == "__main__":
    demonstrate_examples()
    test_fit_to_LD()
