"""
add stress and strain sensor

truss under its own dead weight

solve
    EA u''(x) = - ρgA on Ω

analytical solution
    u(x) = ρgl (x - x ** 2 / 2 / l) / E A
    with Ω = (0, l)
"""
import copy
import dolfin as df
import numpy as np
import pytest
import matplotlib.pylab as plt
from dataclasses import dataclass
from collections import defaultdict

# import matplotlib.pyplot as plt

# MODULE "SENSOR"


class Sensor:
    def measure(self, u, R=None):
        raise NotImplementedError()

    @property
    def name(self):
        return self.__class__.__name__


class DisplacementFieldSensor(Sensor):
    def measure(self, u, R=None):
        return u


class DisplacementSensor(Sensor):
    def __init__(self, where):
        self.where = where

    def measure(self, u, R=None):
        return u(self.where)


class ForceSensor(Sensor):
    def __init__(self, bc):
        self.dofs = list(bc.get_boundary_values().keys())

    def measure(self, u, R=None):
        load_local = np.sum(R[self.dofs])
        return load_local


class StrainFieldSensor(Sensor):
    def __init__(self, kine):
        self.kine = kine  # define function for kinematic eps = kine(u)

    def measure(self, u, R=None):
        # compute strains from displacements
        degree_strain = u.function_space().ufl_element().degree() - 1
        V_eps = df.FunctionSpace(u.function_space().mesh(), "DG", degree_strain)
        eps = df.project(self.kine(u), V_eps)
        return eps


class StrainSensor(Sensor):
    def __init__(self, where, kine):
        self.where = where
        self.kine = kine  # define function for kinematic eps = kine(u)

    def measure(self, u, R=None):
        # compute strains from displacements
        degree_strain = u.function_space().ufl_element().degree() - 1
        V_eps = df.FunctionSpace(u.function_space().mesh(), "DG", degree_strain)
        eps = df.project(self.kine(u), V_eps)
        return eps(self.where)


class StressFieldSensor(Sensor):
    def __init__(self, mat, params):
        self.mat = mat  # define function for material law stress=f(u)
        self.params = params  # parameters like E

    def measure(self, u, R=None):
        # compute strains from displacements
        degree_stress = u.function_space().ufl_element().degree() - 1
        V_stress = df.FunctionSpace(u.function_space().mesh(), "DG", degree_stress)
        stress = df.project(self.mat(u), V_stress)
        return stress


# MODULE "EXPERIMENT"


class Experiment:
    def __init__(self):
        self.data = {}

    def add_sensor_data(self, sensor, data, ts=[1.0]):
        self.data[sensor] = (data, ts)


class UniaxialTrussExperiment(Experiment):
    def __init__(self, parameters):
        super().__init__()
        self.mesh = df.IntervalMesh(1, 0.0, parameters.L)
        self.bcs = None  # to be created externally via .create_bcs

    def create_bcs(self, V):
        def left(x, on_boundary):
            return x[0] < df.DOLFIN_EPS and on_boundary

        self.bcs = [df.DirichletBC(V, df.Constant(0.0), left)]
        return self.bcs

    def update_param(self, parameters):
        self.mesh = df.IntervalMesh(1, 0.0, parameters.L)

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
        self.setup()

    def setup(self):
        mesh = self.experiment.mesh
        self.V = df.FunctionSpace(mesh, "Lagrange", self.params["degree"])
        self.u = df.Function(self.V)
        v = df.TestFunction(self.V)
        self.bcs = self.experiment.create_bcs(self.V)

        rho, g, L, E, A = [
            df.Constant(what)
            for what in [
                self.params.rho,
                self.params.g,
                self.params.L,
                self.params.E,
                self.params.A,
            ]
        ]
        F = rho * g * A
        self.f = df.Expression("t * F", t=0.0, F=F, degree=0)
        self.R = E * df.inner(df.grad(self.u), df.grad(v)) * df.dx - self.f * v * df.dx

    def eps(self, u):
        return u.dx(0)

    def sigma(self, u):
        return self.params.E * u.dx(0)

    def solve(self, t=1.0):
        self.f.t = t
        df.solve(
            self.R == 0,
            self.u,
            self.bcs,
            solver_parameters={"newton_solver": {"relative_tolerance": 1.0e-0}},
        )  # why this tolerance??

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

    def __call__(self, sensors, ts=[1.0]):
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
        value[0] = p.rho * p.g * p.L * (x[0] - x[0] ** 2 / 2 / p.L) / p.E * p.A

    def value_shape(self):
        return ()


class StrainSolution(df.UserExpression):
    def __init__(self, parameters):
        super().__init__(degree=8)
        self.parameters = parameters

    def eval(self, value, x):
        p = self.parameters
        value[0] = p.rho * p.g * p.L * (1 - x[0] / p.L) / p.E * p.A

    def value_shape(self):
        return ()


# EXAMPLE APPLICATION: "CONVERGENCE TEST"
def run_convergence_sjard(
    experiment, parameters, sensor, max_n_refinements=15, eps=1.0e-8
):
    problem = LinearElasticity(experiment, parameters)
    errors = []
    # compute first solution on coarsest mesh
    u_i = problem(sensor)

    experiment.refine()
    problem.setup()

    for n_refinements in range(max_n_refinements):
        u_i1 = problem(sensor)

        # compute relative error
        try:
            # numpy ?
            err = np.linalg.norm(u_i1 - u_i) / np.linalg.norm(u_i1)
            errors.append(err)
        except TypeError:
            scale = df.norm(u_i1, norm_type="l2", mesh=experiment.mesh)
            err = df.errornorm(u_i1, u_i, norm_type="l2", mesh=experiment.mesh) / scale
            errors.append(err)
        if err < eps:
            print(f"----------- CONVERGED -----------")
            print(f" n_refinement = {n_refinements}, Error = {err}")
            break
        u_i = u_i1

        experiment.refine()
        problem.setup()
        n_refinements += 1

    print(
        f"Finally converged. Please use {n_refinements=} with degree {parameters['degree']}."
    )
    return n_refinements


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
            print(f"----------- CONVERGED -----------")
            print(f" n_refinement = {n_refinements}, Error = {err}")
            break
        else:
            print(f"----------- NOT CONVERGED -------")
            print(f" n_refinement = {n_refinements}, Error = {err}")

        experiment.refine()
        n_refinements += 1

    print(
        f"Finally converged. Please use {n_refinements=} with degree {parameters['degree']}."
    )
    return n_refinements


# EXAMPLE APPLICATION: "PARAMETER ESTIMATION"


def estimate(problem, sensor, what):
    from scipy.optimize import least_squares

    param = problem.params.copy()
    value_exp, ts = problem.experiment.data[sensor]

    def error(prm):
        problem.params[what] = prm.squeeze()
        print(f"Try {what} = {problem.params[what]}")
        problem.setup()
        value_fem = problem(sensor, ts)
        return value_fem - value_exp

    optimize_result = least_squares(fun=error, x0=param[what])
    return optimize_result.x.squeeze()


def scalar_parameter_study(
    problem,
    sensor,
    to_vary,
    show=True,
):
    # TODO: 1) how to decide what kind of output is required? 2D/3D plot, points/lines?
    #          maybe somehow define this in the sensor?
    #       2) return only the list and do the plotting "postprocessing" outside the function, add the plot as a flag?
    #       3) what to do when adding time dependency? timestep freuquency as parameters? point -> line, line -> area (3D/colorplot?)
    #       4) currently problem if paramter is not float (eg. dimension). is this a relevant probem?
    #       5) parameters are defined at multiple "levels/places", current workaround experiment.update
    #          does not seem to be the best solution? also 2 calcs crash...
    #          maybe a problem class that gets experiement, material and parameters? instead of passing param to both?

    # validation
    og_params = problem.params.copy()
    # for prm in to_vary:
    # assert prm in og_params

    # solution
    result = defaultdict(list)
    for prm, values in to_vary.items():
        for value in values:
            problem.params[prm] = value
            problem.experiment.update_param(problem.params)
            problem.setup()
            solution = problem(sensor)
            result[prm].append(solution)
        problem.params[prm] = og_params[prm]

    # plotting
    fig, axes = plt.subplots(len(to_vary))
    fig.suptitle(f"Measurements of {sensor.name}")
    fig.tight_layout()
    for ax, prm in zip(axes, to_vary):
        ax.plot(to_vary[prm], result[prm])
        ax.set_xlabel(f"Parameter {prm}")
    if show:
        plt.show()

    return result, (fig, axes)


@dataclass
class Parameters:
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def copy(self):
        return copy.copy(self)


@dataclass
class LinearElasticityParameters(Parameters):
    L: float = 42.0  # length
    A: float = 4.0  # cross section
    E: float = 10.0  # Young's modulus
    g: float = 9.81  # gravitational constant
    rho: float = 7.0  # density
    degree: int = 1  # interpolation degree


def main_convergence_incremental():
    """convergence analysis without analytical solution"""
    parameters = LinearElasticityParameters()
    experiment = get_experiment("UniaxialTruss", parameters)
    full_u_sensor = DisplacementFieldSensor()
    n_refinements = run_convergence_sjard(experiment, parameters, full_u_sensor)
    assert n_refinements == 13


def main_convergence_analytical():
    """convergence analysis using the analytic solution for two kinds of sensors"""
    parameters = LinearElasticityParameters()
    experiment = get_experiment("UniaxialTruss", parameters)

    u_field_sensor = DisplacementFieldSensor()
    u_field_correct = DisplacementSolution(parameters)
    experiment.add_sensor_data(u_field_sensor, u_field_correct)

    u_sensor = DisplacementSensor(where=parameters.L)
    u_correct = u_sensor.measure(u_field_correct)
    experiment.add_sensor_data(u_sensor, u_correct)

    # Run the convergence analysis with the maximum displacement. As for all
    # nodal values, we expect them to be correct even for a single linear
    # element.
    for degree in [1, 2, 3]:
        parameters.degree = degree
        n_refinements = run_convergence(experiment, parameters, u_sensor)
        assert n_refinements == 0

    # Run the convergence analysis with the whole displacement field.
    # Here, a linear solution can only be exact up to a given epsilon
    # Sjard: And therefore assuming a fixed number of refinements makes
    # no sense for the new convergence study
    # Quadratic and cubic interpolation caputure the field without refinement.
    for degree, expected_n_refinements in [(3, 0), (2, 0), (1, 8)]:
        # for degree, expected_n_refinements in [(3, 0), (2, 0)]:
        parameters["degree"] = degree
        n_refinements = run_convergence(experiment, parameters, u_field_sensor, eps=1)
        assert n_refinements == expected_n_refinements


def main_parameter_study():
    """
    Example implementation on how to use the modules for a simple parameter
    study, here the influence of various parameters on the maximum truss
    displacement
    """
    parameters = LinearElasticityParameters()
    experiment = get_experiment("UniaxialTruss", parameters)
    problem = LinearElasticity(experiment, parameters)
    u_sensor = DisplacementSensor(where=parameters["L"])

    scale = np.r_[0.25, 0.5, 1, 2, 4, 8]
    to_vary = {
        "E": parameters["E"] * scale,
        "g": parameters["g"] * scale,
        "degree": [1, 2, 3],
        "A": parameters["A"] * scale,
    }

    scalar_parameter_study(problem, u_sensor, to_vary, show=False)


def main_E_inference():
    parameters = LinearElasticityParameters()
    experiment = get_experiment("UniaxialTruss", parameters)
    problem = LinearElasticity(experiment, parameters)

    u_sensor = DisplacementSensor(where=parameters["L"])
    u_correct = u_sensor.measure(DisplacementSolution(parameters))

    experiment.add_sensor_data(u_sensor, u_correct)

    problem.params.E = 0.42  # some wrong value
    E = estimate(problem, u_sensor, what="E")
    assert E == pytest.approx(parameters.E)


def main_fit_to_LD():
    parameters = LinearElasticityParameters()
    experiment = get_experiment("UniaxialTruss", parameters)
    problem = LinearElasticity(experiment, parameters)
    force_sensor = ForceSensor(problem.bcs[0])

    F = problem(force_sensor, ts=[1.0])
    F_at_t_1 = -parameters.L * parameters.A * parameters.rho * parameters.g
    assert F == pytest.approx(F_at_t_1)

    ts = np.linspace(0, 1, 11)
    F_correct = F_at_t_1 * ts
    experiment.add_sensor_data(force_sensor, F_correct, ts)

    problem.params["A"] = 0.42  # some wrong value
    A = estimate(problem, force_sensor, what="A")
    assert A == pytest.approx(parameters["A"])


def plot_field(problem, sensor, ax=None):
    if ax is None:
        ax = plt.gca()

    def plot_single_field(field, ax, **kwargs):
        x = field.function_space().tabulate_dof_coordinates()
        ax.scatter(x, field.vector().get_local(), **kwargs)

    field = problem(sensor)
    plot_single_field(field, ax, marker="+", label="FEM")

    try:
        compare = problem.experiment.data[sensor][0]
        field.interpolate(compare)
        plot_single_field(field, ax, marker=".", label="ref")
    except KeyError:
        pass
    ax.set_xlabel("x")
    ax.set_ylabel(sensor.name)
    ax.legend()


def main_strain_sensors():
    parameters = LinearElasticityParameters(degree=2)
    experiment = get_experiment("UniaxialTruss", parameters)
    experiment.refine(5)
    problem = LinearElasticity(experiment, parameters)

    # check strain field sensor vs analytic solution
    eps_field_sensor = StrainFieldSensor(kine=problem.eps)
    eps_correct = StrainSolution(parameters)
    assert df.errornorm(eps_correct, problem(eps_field_sensor)) < 1.0e-8

    # check strain sensor vs analytic solution
    eps_max = eps_correct(0)
    eps_sensor = StrainSensor(where=0, kine=problem.eps)
    assert problem(eps_sensor) == pytest.approx(eps_max)

    # create stress sensor for plotting
    stress_field_sensor = StressFieldSensor(mat=problem.sigma, params=parameters)
    experiment.add_sensor_data(eps_field_sensor, eps_correct)

    fig, axes = plt.subplots(2, 1, sharex=True)
    plot_field(problem, eps_field_sensor, axes[0])
    plot_field(problem, stress_field_sensor, axes[1])


if __name__ == "__main__":
    main_convergence_analytical()  # Philipp
    main_convergence_incremental()  # Sjard
    main_parameter_study()  # Erik
    main_E_inference()  # Thomas
    main_fit_to_LD()  # Thomas
    main_strain_sensors()  # Annika
    plt.show()
