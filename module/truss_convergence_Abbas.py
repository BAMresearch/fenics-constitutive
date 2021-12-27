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
    def measure(self, u):
        return u


class DisplacementSensor:
    def __init__(self, where):
        self.where = where

    def measure(self, u):
        return u(self.where)


# MODULE "EXPERIMENT"


class Experiment:
    def __init__(self):
        self.data = {}

    def add_sensor_data(self, sensor, data):
        self.data[sensor] = data


class UniaxialTrussExperiment(Experiment):
    def __init__(self, problem_pars):
        super().__init__()
        self.problem_pars = problem_pars
        self.mesh = df.IntervalMesh(1, 0.0, self.problem_pars["L"])

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


def get_experiment(name, problem_pars):
    # metaprogramming!
    cls_name = name + "Experiment"
    return eval(cls_name)(problem_pars)


# MODULE "PROBLEM"


class LinearElasticity:
    def __init__(self, experiment, model_pars):
        self.experiment = experiment
        self.model_pars = model_pars

    def solve(self):
        mesh = self.experiment.mesh
        V = df.FunctionSpace(mesh, "Lagrange", self.model_pars["degree"])
        u = df.TrialFunction(V)
        v = df.TestFunction(V)
        bcs = self.experiment.create_bcs(V)

        L, A = [
            df.Constant(self.experiment.problem_pars[what]) for what in ["L", "A"]
        ]
        E, rho, g = df.Constant(self.model_pars["E"]), self.model_pars["rho"], self.model_pars["g"]
        f = df.Constant(rho * g * A)
        a = E * df.inner(df.grad(u), df.grad(v)) * df.dx
        L = f * v * df.dx

        solution = df.Function(V)
        df.solve(a == L, solution, bcs)
        return solution

    def __call__(self, sensors):
        """
        Evaluates the problem for the given sensors
        """
        u = self.solve()
        try:
            # only one sensor
            return sensors.measure(u)
        except AttributeError:
            # list of sensors
            return {s: s.measure(u) for s in sensors}


class AReferenceSolution(df.UserExpression):
    """
    In this example, it is an analytical solution.
    But in general, it can be any reference solution, e.g. from a very fine mesh.
    """
    def __init__(self, problem_pars, model_pars):
        super().__init__(degree=8)
        self.problem_pars = problem_pars
        self.model_pars = model_pars

    def eval(self, value, x):
        pp = self.problem_pars
        mp = self.model_pars
        L, A = pp["L"], pp["A"]
        E, rho, g = mp["E"], mp["rho"], mp["g"]
        value[0] = rho * g * L * (x[0] - x[0] ** 2 / 2 / L) / E * A

    def value_shape(self):
        return ()


# EXAMPLE APPLICATION: "CONVERGENCE TEST"


def run_convergence(experiment, model_pars \
                    , sensor, max_n_refinements=15, eps=1.0e-4):
    problem = LinearElasticity(experiment, model_pars)

    for n_refinements in range(max_n_refinements):
        u_fem = problem(sensor)
        u_reference = experiment.data[sensor]

        try:
            # numpy ?
            err = np.linalg.norm(u_fem - u_reference)
        except TypeError:
            err = df.errornorm(u_reference, u_fem, norm_type="l2", mesh=experiment.mesh)

        if err < eps:
            break

        experiment.refine()
        n_refinements += 1

    print(f"Finally converged. Please use {n_refinements=}.")
    return n_refinements


# EXAMPLE APPLICATION: "PARAMETER ESTIMATION"


def estimate_E(experiment, model_pars, sensor):
    from scipy.optimize import minimize_scalar

    def error(E):
        model_pars["E"] = E
        print(f"Try {model_pars['E'] = }")
        problem = LinearElasticity(experiment, model_pars)
        value_fem = problem(sensor)
        value_exp = experiment.data[sensor]
        return abs(value_fem - value_exp)

    optimize_result = minimize_scalar(
        fun=error, bracket=[0.5 * model_pars["E"], 2 * model_pars["E"]], tol=1.0e-8
    )
    return optimize_result.x


if __name__ == "__main__":
    problem_pars = {
        # parameters that prescribe a physical phenomenon regardless of a model associated with it.
        "L": 42.0,
        "A": 4.0,
    }
    model_pars = {
        # parameters that prescribe a model (associated with a problem/experiment).
        "E": 10.0,
        "rho": 7.0,
        "g": 9.81,
        "degree": 1,
    }
    experiment = get_experiment("UniaxialTruss", problem_pars)

    # attach analytic solution for the full displacement field
    full_u_sensor = DisplacementFieldSensor()
    u_reference = AReferenceSolution(problem_pars, model_pars)
    experiment.add_sensor_data(full_u_sensor, u_reference)

    # attach analytic solution for the bottom displacement
    u_sensor = DisplacementSensor(where=problem_pars["L"])
    pp = problem_pars
    mp = model_pars
    u_max = 0.5 * mp["rho"] * mp["g"] * pp["A"] * pp["L"] ** 2 / mp["E"]
    experiment.add_sensor_data(u_sensor, u_max)

    # First assume that we don't know E...
    model_pars["E"] = 1.0
    # ...so we infer it using the measuremnts.
    model_pars["E"] = estimate_E(experiment, model_pars, u_sensor)

    # Run the convergence analysis with the maximum displacement. As for all
    # nodal values, we expect them to be correct even for a single linear
    # element.
    for degree in [1, 2, 3]:
        model_pars["degree"] = degree
        n_refinements = run_convergence(experiment, model_pars, u_sensor)
        assert n_refinements == 0

    # Run the convergence analysis with the whole displacement field.
    # Here, a linear solution can only be exact up to a given epsilon.
    # Quadratic and cubic interpolation caputure the field without refinement.
    for degree, expected_n_refinements in [(3, 0), (2, 0), (1, 14)]:
        model_pars["degree"] = degree
        n_refinements = run_convergence(experiment, model_pars, full_u_sensor)
        assert n_refinements == expected_n_refinements
