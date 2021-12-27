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
        return u(self.where)


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
        lx, ly, x_from, x_to, unit_res = \
            (self.pars[what] for what in ["lx", "ly", "x_from", "x_to", "unit_res"])
        self.mesh = df.RectangleMesh(df.Point(0., 0.), df.Point(lx, ly) \
                                     , int(unit_res * lx), int(unit_res * ly), diagonal='right')
        if x_from is None:
            x_from = 0.0
        if x_to is None:
            x_to = lx
        def _refine_domain(x):
            return df.between(x[0], (x_from - tol, x_to + tol))
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

    def refine(self, N=1):
        """
        Refines the mesh `N` times.
        """
        for i in range(N):
            mf_ref = df.MeshFunction('bool', self.mesh, 2)
            mf_ref.set_all(False)
            df.AutoSubDomain(self._refine_domain).mark(mf_ref, True)
            self.mesh = df.refine(self.mesh, mf_ref)

def get_experiment(name, expr_pars):
    # metaprogramming!
    cls_name = name + "Experiment"
    return eval(cls_name)(expr_pars)

# MODULE "model"

#### to be developed ...


if __name__ == "__main__":
    expr_pars = {
        # Parameters that prescribe a physical phenomenon regardless of a model associated with it.
        "lx": 100.0,
        "ly": 30.0,
        ## The mesh will be refined in the interval [x_from, x_to]
            # If they are given as None, the whole domain is considered for refinement.
        # "x_from": 40,
        # "x_to": 60,
        "x_from": None,
        "x_to": None,
        "unit_res": 0.3,
        ## The support positions. Default (None) implies the supports on very end points.
        "right_sup": None,
        "left_sup": None,
        "x_fix": 'left', # on the left side is fixed in x-direction.
    }
    model_pars = {
        # Parameters that prescribe a model (associated with an experiment).
        "E": 10.0,
        "degree": 1,
        "elem_type": 'CG',
        "dep_dim": 2,
    }
    experiment = get_experiment("Bend3Point2D", expr_pars)
    elem = df.VectorElement(model_pars["elem_type"], experiment.mesh.ufl_cell() \
                            , model_pars["degree"], dim=model_pars["dep_dim"])
    df.FiniteElement('CG', experiment.mesh.ufl_cell(), model_pars["degree"])
    V = df.FunctionSpace(experiment.mesh, elem)
    bcs = experiment.create_bcs(V)
    
    ### ... ongoing ...
