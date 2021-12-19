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
        self.dofs = list(self.bc.get_boundary_values().keys())

    def measure(self, u, R):
        load_local = np.sum(R[self.dofs])
        return load_local


# MODULE "EXPERIMENT"


class Experiment:
    def __init__(self):
        self.data = {}

    def add_sensor_data(self, sensor, data):
        self.data[sensor] = data


class UniaxialTrussExperiment(Experiment):
    def __init__(self, parameters):
        super().__init__()
        self.mesh = df.IntervalMesh(1, 0.0, parameters["L"])

    def create_bcs(self, V):
        def left(x, on_boundary):
            return x[0] < df.DOLFIN_EPS and on_boundary

        return [df.DirichletBC(V, df.Constant(0.0), left)]

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
        f = df.Constant(rho * g * A)
        self.R = E * df.inner(df.grad(self.u), df.grad(v)) * df.dx - f * v * df.dx

    def solve(self):
        df.solve(
            self.R == 0,
            self.u,
            self.bcs,
            solver_parameters={"newton_solver": {"relative_tolerance": 1.0e-1}},
        )  # why??

        return self.u, df.assemble(self.R)

    def __call__(self, sensors):
        """
        Evaluates the problem for the given sensors
        """
        u, R = self.solve()
        try:
            # only one sensor
            return sensors.measure(u, R)
        except AttributeError:
            # list of sensors
            return {s: s.measure(u, R) for s in sensors}


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
        u_fem = problem(sensor)
        u_correct = experiment.data[sensor]

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


def estimate_E(experiment, parameters, sensor):
    from scipy.optimize import minimize_scalar

    def error(E):
        parameters["E"] = E
        print(f"Try {parameters['E'] = }")
        problem = LinearElasticity(experiment, parameters)
        value_fem = problem(sensor)
        value_exp = experiment.data[sensor]
        return abs(value_fem - value_exp)

    optimize_result = minimize_scalar(
        fun=error, bracket=[0.5 * parameters["E"], 2 * parameters["E"]], tol=1.0e-8
    )
    return optimize_result.x
    

def dev():
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
    # bcs = experiment

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
    for degree, expected_n_refinements in [(3, 0), (2, 0), (1, 14)]:
        parameters["degree"] = degree
        n_refinements = run_convergence(experiment, parameters, full_u_sensor)
        assert n_refinements == expected_n_refinements

if __name__ == "__main__":
    demonstrate_examples()
    # dev()
