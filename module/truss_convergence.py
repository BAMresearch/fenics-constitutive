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
from loguru import logger
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
        V_eps = df.TensorFunctionSpace(u.function_space().mesh(), "DG", degree_strain)
        eps = df.project(self.kine(u), V_eps)
        return eps


class StrainSensor(Sensor):
    def __init__(self, where, kine):
        self.where = where
        self.strain_field_sensor = StrainFieldSensor(kine)

    def measure(self, u, R=None):
        return self.strain_field_sensor.measure(u)(self.where)


class StressFieldSensor(Sensor):
    def __init__(self, mat, parameters):
        self.mat = mat  # define function for material law stress=f(u)
        self.parameters = parameters  # parameters like E

    def measure(self, u, R=None):
        # compute strains from displacements
        degree_stress = u.function_space().ufl_element().degree() - 1
        V_stress = df.TensorFunctionSpace(u.function_space().mesh(), "DG", degree_stress)
        stress = df.project(self.mat(u), V_stress)
        return stress


# MODULE "EXPERIMENT"


class Experiment:
    def __init__(self):
        self.data = {}
        self.setup()

    def add_sensor_data(self, sensor, data, ts=[1.0]):
        self.data[sensor] = (data, ts)
    
    def refine(self, N=1):
        """
        Refines the mesh `N` times.
        """
        for _ in range(N):
            self.mesh = df.refine(self.mesh)

    def setup(self):
        raise NotImplementedError()


class UniaxialTrussExperiment(Experiment):
    def __init__(self, parameters):
        self.parameters = parameters
        super().__init__()

    def setup(self):
        self.mesh = df.IntervalMesh(1, 0.0, self.parameters.L)

    def create_bcs(self, V):
        def left(x, on_boundary):
            return x[0] < df.DOLFIN_EPS and on_boundary

        return [df.DirichletBC(V, [df.Constant(0.0)], left)]



class Bending3Point2DExperiment(Experiment):
    def __init__(self, parameters):
        self.parameters = parameters
        super().__init__()

    def setup(self):
        lx, ly = self.parameters.lx, self.parameters.ly
        element_length = self.parameters.element_length
        self.mesh = df.RectangleMesh(df.Point(0., 0.), df.Point(lx, ly) \
                                     , int(lx/element_length), int(ly/element_length), diagonal='crossed')


    def create_bcs(self, V):
        def left_support(x, on_boundary):
            return df.near(x[0], 0) and df.near(x[1], 0.)
            
        def right_support(x, on_boundary):
            return df.near(x[0], self.parameters.lx) and df.near(x[1], 0.)

        bc_left = df.DirichletBC(V, (0,0), left_support, method='pointwise')
        bc_right = df.DirichletBC(V.sub(1), df.Constant(0.0), right_support, method='pointwise')

        return [bc_left, bc_right]
    
    

def get_experiment(name, parameters):
    # metaprogramming!
    cls_name = name + "Experiment"
    return eval(cls_name)(parameters)


# MODULE "PROBLEM"

class LinearElasticity:
    def __init__(self, experiment, parameters):
        self.experiment = experiment
        self.parameters = parameters
        self.setup()

    def setup(self):
        mesh = self.experiment.mesh
        self.V = df.VectorFunctionSpace(mesh, "Lagrange", self.parameters["degree"])
        logger.debug(f"DOFs: {self.V.dim()}")
        self.u = df.Function(self.V)
        v = df.TestFunction(self.V)
        self.bcs = self.experiment.create_bcs(self.V)

        F = self.parameters.rho * self.parameters.g * self.parameters.A
        dim = mesh.geometric_dimension()
        body_force = ["0"] * dim
        body_force[-1] = "t * F"
        # lets (for now) assume that the highest dimension is "downwards"

        self.f = df.Expression(body_force, t=0.0,F=F, degree=0)
        self.R = df.inner(self.eps(v), self.sigma(self.u)) * df.dx
        self.R -= df.dot(self.f, v) * df.dx

    def eps(self, u):
        return df.sym(df.grad(u))

    def sigma(self, u):
        dim = self.experiment.mesh.geometric_dimension()
        E, nu = df.Constant(self.parameters.E), df.Constant(self.parameters.nu)
        mu_ = E / 2. / (1. + nu)
        lambda_ = E * nu /(1.+nu) / (1. - 2. * nu)
        return lambda_ * df.div(u) * df.Identity(dim) + 2 * mu_ * self.eps(u)

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
    def __init__(self, parameters, dim=1):
        self.dim = dim
        self.parameters = parameters
        super().__init__(degree=8)
        
    def eval(self, value, x):
        p = self.parameters
        value[0] = p.rho * p.g * p.L * (x[0] - x[0] ** 2 / 2 / p.L) / p.E * p.A

    def value_shape(self):
        return (self.dim,)


class StrainSolution(df.UserExpression):
    def __init__(self, parameters, dim=1):
        self.dim = dim
        self.parameters = parameters
        super().__init__(degree=8)

    def eval(self, value, x):
        p = self.parameters
        value[0] = p.rho * p.g * p.L * (1 - x[0] / p.L) / p.E * p.A

    def value_shape(self):
        return (self.dim, self.dim)


# EXAMPLE APPLICATION: "CONVERGENCE TEST"
def run_convergence_sjard(
    problem, sensor, max_n_refinements=15, eps=1.0e-8
):
    errors = []
    # compute first solution on coarsest mesh
    u_i = problem(sensor)

    problem.experiment.refine()
    problem.setup()

    for n_refinements in range(max_n_refinements):
        u_i1 = problem(sensor)

        # compute relative error
        try:
            # numpy ?
            err = np.linalg.norm(u_i1 - u_i) / np.linalg.norm(u_i1)
            errors.append(err)
        except TypeError:
            scale = df.norm(u_i1, norm_type="l2")
            err = df.errornorm(u_i1, u_i, norm_type="l2") / scale
            errors.append(err)
        if err < eps:
            logger.debug(f"----------- CONVERGED -----------")
            logger.debug(f" n_refinement = {n_refinements}, Error = {err}")
            break
        else:
            logger.debug(f"----------- NOT CONVERGED -------")
            logger.debug(f" n_refinement = {n_refinements}, Error = {err}")

        u_i = u_i1

        problem.experiment.refine()
        problem.setup()
        n_refinements += 1

    logger.info(
        f"Finally converged. Please use {n_refinements=} with degree {problem.parameters['degree']}."
    )
    return n_refinements


def run_convergence(problem, sensor, max_n_refinements=15, eps=1.0e-4):
    u_correct, ts = problem.experiment.data[sensor]

    for n_refinements in range(max_n_refinements):
        problem.setup()
        u_fem = problem(sensor, ts)

        try:
            # numpy ?
            err = np.linalg.norm(u_fem - u_correct)
        except TypeError:
            err = df.errornorm(u_correct, u_fem, norm_type="l2")

        if err < eps:
            logger.debug(f"----------- CONVERGED -----------")
            logger.debug(f" n_refinement = {n_refinements}, Error = {err}")
            break
        else:
            logger.debug(f"----------- NOT CONVERGED -------")
            logger.debug(f" n_refinement = {n_refinements}, Error = {err}")

        problem.experiment.refine()
        n_refinements += 1

    logger.info(
        f"Finally converged. Please use {n_refinements=} with degree {problem.parameters['degree']}."
    )
    return n_refinements


# EXAMPLE APPLICATION: "PARAMETER ESTIMATION"


def estimate(problem, sensor, what):
    from scipy.optimize import least_squares

    param = problem.parameters.copy()
    value_exp, ts = problem.experiment.data[sensor]

    def error(prm):
        problem.parameters[what] = prm.squeeze()
        logger.debug(f"Try {what} = {problem.parameters[what]}")
        problem.setup()
        value_fem = problem(sensor, ts)
        try:
            np.isfinite(value_fem)
            return value_fem - value_exp
        except TypeError:
            return value_fem.vector()[:] - value_exp.vector()[:]

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
    og_params = problem.parameters.copy()
    # for prm in to_vary:
    # assert prm in og_params

    # solution
    result = defaultdict(list)
    for prm, values in to_vary.items():
        for value in values:
            problem.parameters[prm] = value
            problem.experiment.setup()
            problem.setup()
            solution = problem(sensor)
            result[prm].append(solution)
        problem.parameters[prm] = og_params[prm]

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


class Parameters(dict):
    """
    Dict that also allows to access the parameter
        p["parameter"]
    via the matching attribute
        p.parameter
    to make access shorter
    """

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        assert key in self
        self[key] = value

    def __add__(self, other):
        return Parameters({**self, **other})


def uniaxial_truss_parameters(**kwargs):
    p = Parameters()
    p["L"] = 42.0  # length
    p["A"] = 4.0  # cross section
    return p + kwargs

def bending_parameters(**kwargs):
    p = Parameters()
    p["lx"] = 200.0  
    p["ly"] = 30.0  
    p["element_length"] = 10.
    p["A"] = 4.0  # thickness
    return p + kwargs


def linear_elasticity_parameters(**kwargs):
    p = Parameters()
    p["E"] = 10.0  # Young's modulus
    p["nu"] = 0.0  # Poisson's ratio
    p["g"] = 9.81  # gravitational constant
    p["rho"] = 7.0  # density
    p["degree"] = 1  # interpolation degree

    return p + kwargs


def main_convergence_incremental():
    """convergence analysis without analytical solution"""
    parameters = linear_elasticity_parameters() + uniaxial_truss_parameters()
    experiment = get_experiment("UniaxialTruss", parameters)
    problem = LinearElasticity(experiment, parameters)
    full_u_sensor = DisplacementFieldSensor()
    n_refinements = run_convergence_sjard(problem, full_u_sensor)
    assert n_refinements == 13


def main_convergence_analytical():
    """convergence analysis using the analytic solution for two kinds of sensors"""
    parameters = linear_elasticity_parameters() + uniaxial_truss_parameters()
    experiment = get_experiment("UniaxialTruss", parameters)
    problem = LinearElasticity(experiment, parameters)

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
        n_refinements = run_convergence(problem, u_sensor)
        assert n_refinements == 0

    # Run the convergence analysis with the whole displacement field.
    # Here, a linear solution can only be exact up to a given epsilon
    # Sjard: And therefore assuming a fixed number of refinements makes
    # no sense for the new convergence study
    # Quadratic and cubic interpolation caputure the field without refinement.
    for degree, expected_n_refinements in [(3, 0), (2, 0), (1, 8)]:
        # for degree, expected_n_refinements in [(3, 0), (2, 0)]:
        parameters["degree"] = degree
        n_refinements = run_convergence(problem, u_field_sensor, eps=1)
        assert n_refinements == expected_n_refinements


def main_parameter_study():
    """
    Example implementation on how to use the modules for a simple parameter
    study, here the influence of various parameters on the maximum truss
    displacement
    """
    parameters = linear_elasticity_parameters() + uniaxial_truss_parameters()
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
    parameters = linear_elasticity_parameters() + uniaxial_truss_parameters()
    experiment = get_experiment("UniaxialTruss", parameters)
    problem = LinearElasticity(experiment, parameters)

    u_sensor = DisplacementSensor(where=parameters["L"])
    u_correct = u_sensor.measure(DisplacementSolution(parameters))

    experiment.add_sensor_data(u_sensor, u_correct)

    problem.parameters.E = 0.42  # some wrong value
    E = estimate(problem, u_sensor, what="E")
    assert E == pytest.approx(parameters.E)


def main_fit_to_LD():
    parameters = linear_elasticity_parameters() + uniaxial_truss_parameters()
    experiment = get_experiment("UniaxialTruss", parameters)
    problem = LinearElasticity(experiment, parameters)
    force_sensor = ForceSensor(problem.bcs[0])

    F = problem(force_sensor, ts=[1.0])
    F_at_t_1 = -parameters.L * parameters.A * parameters.rho * parameters.g
    assert F == pytest.approx(F_at_t_1)

    ts = np.linspace(0, 1, 11)
    F_correct = F_at_t_1 * ts
    experiment.add_sensor_data(force_sensor, F_correct, ts)

    problem.parameters["A"] = 0.42  # some wrong value
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
    parameters = linear_elasticity_parameters(degree=2) + uniaxial_truss_parameters()
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
    stress_field_sensor = StressFieldSensor(mat=problem.sigma, parameters=parameters)
    experiment.add_sensor_data(eps_field_sensor, eps_correct)

    fig, axes = plt.subplots(2, 1, sharex=True)
    plot_field(problem, eps_field_sensor, axes[0])
    plot_field(problem, stress_field_sensor, axes[1])

def main_bending():
    parameters = linear_elasticity_parameters(degree=2, E=6174) + bending_parameters(element_length=10.)
    experiment = get_experiment("Bending3Point2D", parameters)
    problem = LinearElasticity(experiment, parameters)

    sensor = DisplacementFieldSensor()
    reference_solution = problem(sensor)
    experiment.add_sensor_data(sensor, reference_solution)

    parameters.E = 0.42
    E = estimate(problem, sensor, what="E")
    assert E == pytest.approx(6174)

if __name__ == "__main__":
    main_bending() # Abbas
    main_convergence_analytical()  # Philipp
    main_convergence_incremental()  # Sjard
    main_parameter_study()  # Erik
    main_E_inference()  # Thomas
    main_fit_to_LD()  # Thomas
    main_strain_sensors()  # Annika
    plt.show()
