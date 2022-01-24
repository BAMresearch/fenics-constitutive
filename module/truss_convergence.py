"""
add stress and strain sensor

truss under its own dead weight

solve
    EA u''(x) = - ρgA on Ω

analytical solution
    u(x) = ρgl (x - x ** 2 / 2 / l) / E A
    with Ω = (0, l)
"""
import dolfin as df
import numpy as np
import pytest
import matplotlib.pylab as plt
# import matplotlib.pyplot as plt

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

class StrainFieldSensor:
    def __init__(self,kine):
        self.kine = kine  # define function for kinematic eps = kine(u)

    def measure(self, u, R):
        # compute strains from displacements
        degree_strain = u.function_space().ufl_element().degree()-1
        V_eps = df.FunctionSpace(u.function_space().mesh(), "DG", degree_strain)
        eps = df.project(self.kine(self,u),V_eps)
        return eps

class StrainSensor:
    def __init__(self, where, kine):
        self.where = where
        self.kine = kine  # define function for kinematic eps = kine(u)

    def measure(self, u, R):
        # compute strains from displacements
        degree_strain = u.function_space().ufl_element().degree() - 1
        V_eps = df.FunctionSpace(u.function_space().mesh(), "DG", degree_strain)
        eps = df.project(self.kine(self,u), V_eps)
        return eps(self.where)

class StressFieldSensor:
    def __init__(self,mat,params):
        self.mat = mat  # define function for material law stress=f(u)
        self.params = params # parameters like E

    def measure(self, u, R):
        # compute strains from displacements
        degree_stress = u.function_space().ufl_element().degree()-1
        V_stress = df.FunctionSpace(u.function_space().mesh(), "DG", degree_stress)
        stress = df.project(self.mat(self,u,self.params),V_stress)
        return stress


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
        self.setup()

    def setup(self):
        mesh = self.experiment.mesh
        self.V = df.FunctionSpace(mesh, "Lagrange", self.params["degree"])
        self.u = df.Function(self.V)
        v = df.TestFunction(self.V)
        self.bcs = self.experiment.create_bcs(self.V)

        rho, g, L, E, A = [
            df.Constant(self.params[what]) for what in ["rho", "g", "L", "E", "A"]
        ]
        F = rho* g * A
        self.f = df.Expression("t * F", t=0.0, F=F, degree=0)
        self.R = E * df.inner(df.grad(self.u), df.grad(v)) * df.dx - self.f * v * df.dx


    def eps(self,u):
        return u.dx(0)

    def sigma(self,u,params):
        return params['E']*u.dx(0)

    def solve(self, t=1.):
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

class StrainSolution(df.UserExpression):
    def __init__(self, parameters):
        super().__init__(degree=8)
        self.parameters = parameters

    def eval(self, value, x):
        p = self.parameters
        rho, g, L, E, A = p["rho"], p["g"], p["L"], p["E"], p["A"]
        value[0] = rho * g * L * (1 - x[0] / L) / E * A

    def value_shape(self):
        return ()


# EXAMPLE APPLICATION: "CONVERGENCE TEST"
def run_convergence_sjard(experiment, parameters, sensor, max_n_refinements=15, eps=1.0e-8):
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
            err = np.linalg.norm(u_i1 - u_i)/np.linalg.norm(u_i1)
            errors.append(err)
        except TypeError:
            scale = df.norm(u_i1, norm_type="l2", mesh=experiment.mesh)
            err = df.errornorm(u_i1, u_i, norm_type="l2", mesh=experiment.mesh)/scale
            errors.append(err)
        if err < eps:
            print(f"----------- CONVERGED -----------")
            print(f" n_refinement = {n_refinements}, Error = {err}")
            break
        u_i = u_i1
        
        experiment.refine()
        problem.setup()
        n_refinements += 1

    print(f"Finally converged. Please use {n_refinements=} with degree {parameters['degree']}.")
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

    print(f"Finally converged. Please use {n_refinements=} with degree {parameters['degree']}.")
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



def compare(experiment, parameters, sensor):
    problem = LinearElasticity(experiment, parameters)
    correct, ts = experiment.data[sensor]
    fem = problem(sensor, ts)

    try:
        # numpy ?
        err = np.linalg.norm(fem - correct)
    except (TypeError, AttributeError):
        err = df.errornorm(correct, fem, norm_type="l2", mesh=experiment.mesh)

    return err


# Question: would it be better to "feed" a problem instead of experiment + parameters?
def run_paramterstudy(experiment, parameters, sensor, param_list = None, scale_list = [0.25, 0.5, 2.0, 3.0], show=True):
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
                problem.setup()
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
    if show:
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

    # run sjards convergence study without experimental data!
    parameters["degree"] = 1
    n_refinements = run_convergence_sjard(experiment, parameters, full_u_sensor)
    assert n_refinements == 13



    # attach the analytic solution and 
    experiment.add_sensor_data(full_u_sensor, u_correct)

    # attach analytic solution for the bottom displacement
    u_sensor = DisplacementSensor(where=parameters["L"])
    p = parameters
    u_max = 0.5 * p["rho"] * p["g"] * p["A"] * p["L"] ** 2 / p["E"]
    experiment.add_sensor_data(u_sensor, u_max)


    # Run the paramterstudy
    run_paramterstudy(experiment, parameters, u_sensor, param_list=['E','g','L','degree'], show=False)
  
    
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
    for degree, expected_n_refinements in [(3, 0), (2, 0), (1, 8)]:
    # for degree, expected_n_refinements in [(3, 0), (2, 0)]:
        parameters["degree"] = degree
        n_refinements = run_convergence(experiment, parameters, full_u_sensor, eps=1)
        assert n_refinements == expected_n_refinements

    
    test_fit_to_LD()
    ### added

    # compare strain to given analytic one for fixed refinement
    parameters["degree"] = 2
    experiment.refine(N=0)
    # add one point
    eps_sensor = StrainSensor(where=0,kine=LinearElasticity.eps)
    eps_max =p["rho"] * p["g"] * p["L"] * p["A"] / p["E"]  # add analytic eps at sensor x=0
    experiment.add_sensor_data(eps_sensor, eps_max)

    parameters["degree"] = 2
    experiment.refine(N=0)
    err = compare(experiment, parameters, eps_sensor)
    assert err <= 0.01

    # over whole truss
    full_eps_sensor = StrainFieldSensor(kine=LinearElasticity.eps)
    eps_correct = StrainSolution(parameters)
    experiment.add_sensor_data(full_eps_sensor, eps_correct)
    err = compare(experiment, parameters, full_eps_sensor)
    assert err <= 0.01

    # compute stresses whole field
    # stress sensor
    full_stress_sensor = StressFieldSensor(mat=LinearElasticity.sigma,params=parameters)
    problem = LinearElasticity(experiment, parameters)
    stress_fem = problem(full_stress_sensor)

    # some plotting stuff
    fig, axes = plt.subplots(1, 3)
    ax = axes[0]
    u_fem = problem(full_u_sensor)
    u_correct = df.project(experiment.data[full_u_sensor][0],u_fem.function_space())
    ax.plot(experiment.mesh.coordinates(), u_fem.compute_vertex_values(),'*b',label='u fem')
    ax.plot(experiment.mesh.coordinates(), u_correct.compute_vertex_values(), '+g', label='u correct')
    ax.legend(loc='best')
    ax.set_xlabel('x')
    ax.set_ylabel('u')

    ax = axes[1]
    eps_fem = problem(full_eps_sensor)
    eps_correct = df.project(experiment.data[full_eps_sensor][0],eps_fem.function_space())
    ax.plot(experiment.mesh.coordinates(), eps_fem.compute_vertex_values(), '*b', label='eps fem')
    ax.plot(experiment.mesh.coordinates(), eps_correct.compute_vertex_values(), '+g', label='eps correct')
    ax.legend(loc='best')
    ax.set_xlabel('x')
    ax.set_ylabel('eps')

    ax = axes[2]
    ax.plot(eps_fem.compute_vertex_values(), stress_fem.compute_vertex_values(), '*b', label='stress-strain fem')
    ax.legend(loc='best')
    ax.set_xlabel('strain')
    ax.set_ylabel('stress')

    plt.show()

