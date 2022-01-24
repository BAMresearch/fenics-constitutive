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
import matplotlib.pyplot as plt

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
    def __init__(self, parameters):
        super().__init__()
        self.mesh = df.IntervalMesh(1, 0.0, parameters["L"])

    def create_bcs(self, V):
        def left(x, on_boundary):
            return x[0] < df.DOLFIN_EPS and on_boundary

        return [df.DirichletBC(V, df.Constant(0.0), left)]

    def update_param(self, parameters):
        self.mesh = df.IntervalMesh(1, 0.0, parameters["L"])

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

    def solve(self):
        mesh = self.experiment.mesh
        V = df.FunctionSpace(mesh, "Lagrange", self.params["degree"])
        u = df.TrialFunction(V)
        v = df.TestFunction(V)
        bcs = self.experiment.create_bcs(V)

        rho, g, L, E, A = [
            df.Constant(self.params[what]) for what in ["rho", "g", "L", "E", "A"]
        ]
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


def run_convergence(experiment, parameters, sensor, max_n_refinements=15, eps=1.0e-8):
    problem = LinearElasticity(experiment, parameters)
    errors = []    
    # compute first solution on coarsest mesh
    u_i = problem(sensor)

    experiment.refine()
    
    for n_refinements in range(max_n_refinements):
        u_i1 = problem(sensor)
        
        # compute relative error
        try:
            # numpy ?
            err = np.linalg.norm(u_i1 - u_i)/np.linalg.norm(u_i1)
            errors.append(err)
        except TypeError:
            scale = df.norm(u_i1, norm_type="l2", mesh=experiment.mesh)
            err = df.errornorm(u_i1, u_i, norm_type="l2", mesh=experiment.mesh)/scale
            errors.append(err)
        if err < eps:
            # uncomment for error plot
            # import matplotlib.pyplot as plt
            # plt.plot(list(range(n_refinements+1)),errors)
            # plt.yscale("log")
            # plt.show()
            break
        u_i = u_i1
        
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


# Question: would it be better to "feed" a problem instead of experiment + parameters?
def run_paramterstudy(experiment, parameters, sensor, param_list = None, scale_list = [0.25, 0.5, 2.0, 3.0]):
    # TODO: 1) how to decide what kind of output is required? 2D/3D plot, points/lines?
    #          maybe somehow define this in the sensor?
    #       2) return only the list and do the plotting "postprocessing" outside the function, add the plot as a flag?
    #       3) what to do when adding time dependency? timestep freuquency as parameters? point -> line, line -> area (3D/colorplot?)
    #       4) currently problem if paramter is not float (eg. dimension). is this a relevant probem?
    #       5) parameters are defined at multiple "levels/places", current workaround experiment.update
    #          does not seem to be the best solution? also 2 calcs crash...
    #          maybe a problem class that gets experiement, material and parameters? instead of passing param to both?

    # Problem: parameters are somewho defined at multiple places..., how to change geometry paramters...

    # setup problem
    problem = LinearElasticity(experiment, parameters)

    # generate list with paramters to analyze
    if not param_list:
        param_list = list(parameters.keys())
    else:
        # check if given name matches parameter, otherwise remove from list
        delete = []
        for item in param_list:
            if not item in parameters:
                print(f'AAAAHHHHHH {item} is not a valid parameter name!!!')
                delete.append(item)
        for item in delete:
            param_list.remove(item)

    # make a copy of the original parameter dictionary
    og_params = parameters.copy()

    # compute "standard" solution, to compare for all variations
    u_fem = problem(sensor)

    # createlist for results
    plot_results = []
    for key in param_list:
        # add original calculation to all list
        plot_results.append([[parameters[key],u_fem]])

    # setup fiure
    fig, axs = plt.subplots(len(param_list))
    fig.suptitle('Paramterstudy of various paramters')
    fig.tight_layout()

    # loop over paramters and scale factors
    for i, parameter in enumerate(param_list):
        print(f'Computing parameter: {parameter}')

        for scale in scale_list:
            # adjust paramter in problem
            problem.params[parameter] = og_params[parameter] * scale

            experiment.update_param(problem.params)
            try:
                # maybe problem crashes because of strange parameter or other problem
                u_fem = problem(sensor)
                plot_results[i].append([problem.params[parameter],u_fem])
            except:
                # better error message and print which value caused the crash
                print(f'Computation crashed for {parameter} = {problem.params[parameter]}')

        # sort plot list
        plot_results[i].sort()
        # reorder data points, such that a x and a y list is given to plot
        axs[i].plot(*zip(*plot_results[i]))
        # set plot title
        axs[i].set_title(parameter)

        # reset the parameter to original value
        problem.params[parameter] = og_params[parameter]

    # show plot
    plt.show()
    return plot_results










if __name__ == "__main__":
    # define experimental paramters
    # this incudes the strucutral as well as the material paramters
    parameters = {
        "L": 42.0,
        "E": 10.0,
        "g": 9.81,
        "A": 4.0,
        "rho": 7.0,
        "degree": 1,
    }
    # experiement class
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


    # Run the paramterstudy
    run_paramterstudy(experiment, parameters, u_sensor, param_list=['E','g','L','degree'])
    
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
    # Here, a linear solution can only be exact up to a given epsilon
    # Sjard: And therefore assuming a fixed number of refinements makes
    # no sense for the new convergence study
    # Quadratic and cubic interpolation caputure the field without refinement.
    for degree, expected_n_refinements in [(3, 0), (2, 0)]:
        parameters["degree"] = degree
        n_refinements = run_convergence(experiment, parameters, full_u_sensor)
        assert n_refinements == expected_n_refinements

    # Run convergence study for elements of order 1
    # Here we just want to check, that at least one 
    # refinement step is done
    parameters["degree"] = 1
    n_refinements = run_convergence(experiment, parameters, full_u_sensor)
    assert n_refinements >= 1
