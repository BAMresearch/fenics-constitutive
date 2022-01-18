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

# MODULE "SENSOR"


class DisplacementFieldSensor:
    def measure(self, u):
        return u

class DisplacementSensor:
    def __init__(self, where):
        self.where = where

    def measure(self, u):
        return u(self.where)


class StrainFieldSensor:
    def __init__(self,kine):
        self.kine = kine  # define function for kinematic eps = kine(u)

    def measure(self, u):
        # compute strains from displacements
        degree_strain = u.function_space().ufl_element().degree()-1
        V_eps = df.FunctionSpace(u.function_space().mesh(), "DG", degree_strain)
        eps = df.project(self.kine(self,u),V_eps)
        return eps

class StrainSensor:
    def __init__(self, where, kine):
        self.where = where
        self.kine = kine  # define function for kinematic eps = kine(u)

    def measure(self, u):
        # compute strains from displacements
        degree_strain = u.function_space().ufl_element().degree() - 1
        V_eps = df.FunctionSpace(u.function_space().mesh(), "DG", degree_strain)
        eps = df.project(self.kine(self,u), V_eps)
        return eps(self.where)

class StressFieldSensor:
    def __init__(self,mat,params):
        self.mat = mat  # define function for material law stress=f(u)
        self.params = params # parameters like E

    def measure(self, u):
        # compute strains from displacements
        degree_stress = u.function_space().ufl_element().degree()-1
        V_stress = df.FunctionSpace(u.function_space().mesh(), "DG", degree_stress)
        stress = df.project(self.mat(self,u,self.params),V_stress)
        return stress


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

    def eps(self,u):
        return u.dx(0)

    def sigma(self,u,params):
        return params['E']*u.dx(0)

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


def run_convergence(experiment, parameters, sensor, max_n_refinements=15, eps=1.0e-4):
    problem = LinearElasticity(experiment, parameters)

    for n_refinements in range(max_n_refinements):
        u_fem = problem(sensor)
        u_correct = experiment.data[sensor]


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

def compare(experiment, parameters, sensor):
    problem = LinearElasticity(experiment, parameters)
    fem = problem(sensor)
    correct = experiment.data[sensor]

    try:
        # numpy ?
        err = np.linalg.norm(fem - correct)
    except TypeError:
        err = df.errornorm(correct, fem, norm_type="l2", mesh=experiment.mesh)

    return err


if __name__ == "__main__":
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
    for degree, expected_n_refinements in [(3, 0), (2, 0), (1, 15)]:
        parameters["degree"] = degree
        n_refinements = run_convergence(experiment, parameters, full_u_sensor)
        assert n_refinements == expected_n_refinements


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
    experiment.add_sensor_data(full_stress_sensor, None)
    problem = LinearElasticity(experiment, parameters)
    stress_fem = problem(full_stress_sensor)

    # some plotting stuff
    import matplotlib.pylab as plt

    fig, axes = plt.subplots(1, 3)
    ax = axes[0]
    u_fem = problem(full_u_sensor)
    u_correct = df.project(experiment.data[full_u_sensor],u_fem.function_space())
    ax.plot(experiment.mesh.coordinates(), u_fem.compute_vertex_values(),'*b',label='u fem')
    ax.plot(experiment.mesh.coordinates(), u_correct.compute_vertex_values(), '+g', label='u correct')
    ax.legend(loc='best')
    ax.set_xlabel('x')
    ax.set_ylabel('u')

    ax = axes[1]
    eps_fem = problem(full_eps_sensor)
    eps_correct = df.project(experiment.data[full_eps_sensor],eps_fem.function_space())
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

