"""
A 3-point bending of 2-D
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
        return u(self.where)[1] # in y-direction

class ForceSensor:
    def __init__(self, where):
        self.where = where

    def measure(self, R): # R: residual
        return None # to be developed

# MODULE "EXPERIMENT"

class Experiment:
    def __init__(self):
        self.data = {}

    def add_sensor_data(self, sensor, data):
        self.data[sensor] = data

class Bend3Point2DExperiment(Experiment):
    def __init__(self, expr_pars):
        super().__init__()
        self.pars = expr_pars
        tol = 1e-14
        lx, ly, refine_from, refine_to, unit_res = \
            (self.pars[what] for what in ["lx", "ly", "refine_from", "refine_to", "unit_res"])
        self.mesh = df.RectangleMesh(df.Point(0., 0.), df.Point(lx, ly) \
                                     , int(unit_res * lx), int(unit_res * ly), diagonal='right')
        if refine_from is None:
            refine_from = 0.0
        if refine_to is None:
            refine_to = lx
        def _refine_domain(x):
            return df.between(x[0], (refine_from - tol, refine_to + tol))
        self._refine_domain = _refine_domain

    def create_bcs(self, V):
        tol = 1e-14
        right_sup = self.pars["right_sup"]
        left_sup = self.pars["left_sup"]
        if right_sup is None:
            right_sup = self.pars["lx"]
        if left_sup is None:
            left_sup = 0.0
        x_fix = self.pars["x_fix"]
        
        bcs = []
        # It is crucial to not include on_boundary for pointwise boundary condition
        def left_bot(x, on_boundary):
            return df.near(x[0], left_sup, tol) and df.near(x[1], 0., tol)
        bc_left_y = df.DirichletBC(V.sub(1), df.Constant(0.0), left_bot, method='pointwise')
        if len(bc_left_y.get_boundary_values()) == 0:
            raise ValueError('No DOFs were found for the left-side support. You might need to redefine the mesh.')
        bcs.append(bc_left_y)
        
        def right_bot(x, on_boundary):
            return df.near(x[0], right_sup, tol) and df.near(x[1], 0., tol)
        bc_right_y = df.DirichletBC(V.sub(1), df.Constant(0.0), right_bot, method='pointwise')
        if len(bc_right_y.get_boundary_values()) == 0:
            raise ValueError('No DOFs were found for the right-side support. You might need to redefine the mesh.')
        bcs.append(bc_right_y)
        
        if x_fix=='left':
            bc_left_x = df.DirichletBC(V.sub(0), df.Constant(0.0), left_bot, method='pointwise')
            if len(bc_left_x.get_boundary_values()) == 0:
                raise ValueError('No DOFs were found for the left-side support. You might need to redefine the mesh.')
            bcs.append(bc_left_x)
        else: # means on right
            bc_right_x = df.DirichletBC(V.sub(0), df.Constant(0.0), right_bot, method='pointwise')
            if len(bc_right_x.get_boundary_values()) == 0:
                raise ValueError('No DOFs were found for the right-side support. You might need to redefine the mesh.')
            bcs.append(bc_right_x)
        return bcs
    
    def create_loads(self, V):
        """
        The output is a list of load-expressions and a list of BCs (possibly) associated with them.
        
        ???:
        We might need to restrict loads to be - for sure - time-dependent (even in case of constant load)
            , where the time is either quasi-static or dynamic and each load expression's time
            can be accessed via load.t, which can be evolved in a TimeStepper.
        """
        tol = 1e-14
        x1 = self.pars["load_from"]
        x2 = self.pars["load_to"]
        f = self._get_disp_load()
        ly = self.pars["ly"]
        ## on an interval
        def middle_top(x, on_boundary):
            return on_boundary and df.between(x[0], (x1 - tol, x2 + tol)) and df.near(x[1], ly, tol)
        bc_middle_top = df.DirichletBC(V.sub(1), f, middle_top)
        if len(bc_middle_top.get_boundary_values()) == 0:
            raise ValueError('No DOFs were found for the middle displacement loading. You might need to redefine the mesh.')
        return [bc_middle_top], [f]
    
    def _get_disp_load(self):
        """
        This can be extended to a time-dependent load.
        """
        _degree = 1
        f_max = self.pars["disp_load"]
        t0 = 1.0
        return df.Expression('f_max * t / T', degree=_degree, t=t0, T=1.0, f_max=f_max)

    def refine(self, N=1):
        """
        Refines the mesh `N` times.
        """
        for i in range(N):
            mf_ref = df.MeshFunction('bool', self.mesh, 2)
            mf_ref.set_all(False)
            df.AutoSubDomain(self._refine_domain).mark(mf_ref, True)
            self.mesh = df.refine(self.mesh, mf_ref)
    
    @property
    def dxm(self):
        return df.dx(self.mesh)

def get_experiment(name, expr_pars):
    # metaprogramming!
    cls_name = name + "Experiment"
    return eval(cls_name)(expr_pars)


# MODULE "model"

def dep_dim(constraint): # Returns proper dimension (vector size) for dependent variable
    constraint_switcher = {
                'UNIAXIAL': 1,
                'PLANE_STRESS': 2,
                'PLANE_STRAIN': 2,
                '3D': 3
             }
    _dim = constraint_switcher.get(constraint, "Invalid constraint given. Possible values are: " + str(constraint_switcher.keys()))
    return _dim

def hooke_coeffs(E, nu, constraint):
    lamda=(E*nu/(1+nu))/(1-2*nu)
    mu=E/(2*(1+nu))
    if constraint=='PLANE_STRESS':
        lamda = 2*mu*lamda/(lamda+2*mu)
    return mu, lamda

class ElasticConstitutive():
    def __init__(self, E, nu, constraint):
        self.E = E
        self.nu = nu
        self.constraint = constraint
        self.dim = dep_dim(self.constraint)
        if self.dim != 1:
            self.mu, self.lamda = hooke_coeffs(E=self.E, nu=self.nu, constraint=constraint)
    def sigma(self, eps):
        if self.dim == 1:
            return self.E * eps
        else:
            return self.lamda * df.tr(eps) * df.Identity(self.dim) + 2 * self.mu * eps

class LinearElasticity2D:
    def __init__(self, experiment, model_pars):
        self.experiment = experiment
        self.model_pars = model_pars
        self.build()

    def build(self):
        _constraint = self.model_pars["constraint"]
        _dim = dep_dim(_constraint)
        if _dim==1:
            elem = df.FiniteElement(self.model_pars["elem_type"], self.experiment.mesh.ufl_cell() \
                            , self.model_pars["degree"])
        else:
            elem = df.VectorElement(self.model_pars["elem_type"], self.experiment.mesh.ufl_cell() \
                            , self.model_pars["degree"], dim=_dim)
        V = df.FunctionSpace(self.experiment.mesh, elem)
        self.U = df.Function(V)
        v = df.TestFunction(V)
        
        E, nu = df.Constant(self.model_pars["E"]), self.model_pars["nu"]
        self.mat = ElasticConstitutive(E, nu, _constraint)
        eps = df.sym(df.grad(self.U))
        sig = self.mat.sigma(eps)
        eps_v = df.sym(df.grad(v))
        f = df.Constant(_dim * (0.0,))
        dxm = self.experiment.dxm
        a_u = df.inner(sig, eps_v) * dxm
        L_u = df.inner(f, v) * dxm
        self.F = a_u - L_u
        
        bcs0 = self.experiment.create_bcs(V)
        bcs1, loads = self.experiment.create_loads(V)
        self.bcs = bcs0 + bcs1
    
    def solve(self):
        df.solve(self.F==0.0, self.U, self.bcs)
        return self.U

    def __call__(self, sensors, rebuild=False):
        """
        Evaluates the model for the given sensors
        """
        if rebuild:
            self.build()
        u = self.solve()
        try:
            # only one sensor
            return sensors.measure(u)
        except AttributeError:
            # list of sensors
            return {s: s.measure(u) for s in sensors}

class AReferenceSolution():
    """
    In this example, it is from a very fine mesh.
    """
    def __init__(self, expr_pars, model_pars):
        experiment = get_experiment("Bend3Point2D", expr_pars) # we need a fresh experiment, since the mesh will be refined.
        experiment.refine(1) # quite refined
        import copy
        model_pars = copy.deepcopy(model_pars)
        model_pars["degree"] = 6 # a high interpolation degree for the reference solution
        model = LinearElasticity2D(experiment, model_pars)
        self.U = model.solve() # a FEniCS function containing the reference solution

def run_convergence(experiment, model_pars \
                    , sensor, max_n_refinements=10, eps=1.0e-2):
    model = LinearElasticity2D(experiment, model_pars)
    _conv = False
    for n_refinements in range(max_n_refinements):
        u_fem = model(sensor, rebuild=True) # we must rebuild the model due to refining mesh.
        u_reference = experiment.data[sensor]

        try:
            # numpy ?
            err = np.linalg.norm(u_fem - u_reference)
        except TypeError:
            err = df.errornorm(u_reference, u_fem, norm_type="l2", mesh=experiment.mesh)

        if err < eps:
            print(f"----------- CONVERGED -----------")
            print(f" n_refinement = {n_refinements}, Error = {err}")
            _conv = True
            break
        else:
            print(f"----------- NOT CONVERGED -------")
            print(f" n_refinement = {n_refinements}, Error = {err}")

        experiment.refine()
        n_refinements += 1
    if _conv:
        print(f"Finally converged. Please use {n_refinements=}.")
    else:
         print(f"Not converged even after maximum number of refinements (={max_n_refinements}).")
    return n_refinements

def estimate_E(experiment, model_pars, sensor):
    from scipy.optimize import minimize_scalar

    def error(E):
        model_pars["E"] = E
        print(f"Try {model_pars['E'] = }")
        model = LinearElasticity2D(experiment, model_pars)
        value_fem = model(sensor)
        value_exp = experiment.data[sensor]
        return abs(value_fem - value_exp)

    optimize_result = minimize_scalar(
        fun=error, bracket=[0.5 * model_pars["E"], 2 * model_pars["E"]], tol=1.0e-8
    )
    return optimize_result.x

if __name__ == "__main__":
    expr_pars = {
        ### Parameters that prescribe a physical phenomenon regardless of a model associated with it.
        ## GEOMETRY
        "lx": 100.0,
        "ly": 20.0,
        ## BCs (SUPPORTs)
        # The support positions. Default (None) implies the supports on very end points.
        "right_sup": None,
        "left_sup": None,
        "x_fix": 'left', # on the left side is fixed in x-direction.
        ## LOADs (and possible associated BCs)
        # The load is displacement-controlled and applied on top edge in the interval [load_from, load_to]
        "load_from": 40,
        "load_to": 60,
        "disp_load": -3, # negative sign implies in (-y) direction
        ## MESH
        # The mesh will be refined in the interval [refine_from, refine_to]
            # If they are given as None, the whole domain is considered for refinement.
        # "refine_from": 40,
        # "refine_to": 60,
        "refine_from": None,
        "refine_to": None,
        "unit_res": 0.1, # number of mesh per length
    }
    model_pars = {
        # Parameters that prescribe a model (associated with an experiment).
        "E": 10.0,
        "nu": 0.2,
        "constraint": 'PLANE_STRESS',
        "degree": 1,
        "elem_type": 'CG',
    }
    experiment = get_experiment("Bend3Point2D", expr_pars)
    
    # Attach reference solution for the full displacement field
    sensor_u_full = DisplacementFieldSensor()
    u_reference = AReferenceSolution(expr_pars, model_pars).U
    experiment.add_sensor_data(sensor_u_full, u_reference)
    
    #### CONVERGENCE ####
        # Run the convergence analysis with the whole displacement field.
    n_refinements = run_convergence(experiment, model_pars, sensor_u_full)
    
    #### INFERENCE ####
    # Attach reference solution for the beam deflection (at middle top)
    _where = [expr_pars["lx"]/2.0, expr_pars["ly"]]
    # sensor_f = ForceSensor(where=_where)
    # f_ref = sensor_f(_where)
    # experiment.add_sensor_data(sensor_f, f_ref)
    # E_target = model_pars["E"]
    # model_pars["E"] = 1.0
    # E_estimated = estimate_E(experiment, model_pars, u_sensor)
    
    
