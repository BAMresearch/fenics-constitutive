#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 2021

@author: ajafari
"""

import os, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
import dolfin as df
from dolfin import Function as df_Function
from dolfin import info as df_info
tol = 1e-14

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def boundary_condition(i, u, x):
    bc = df.DirichletBC(i, u, x)
    bc_dofs = [key for key in bc.get_boundary_values().keys()]
    return bc, bc_dofs

def boundary_condition_pointwise(i, u, x):
    bc = df.DirichletBC(i, u, x, method='pointwise')
    bc_dofs = [key for key in bc.get_boundary_values().keys()]
    return bc, bc_dofs

def load_and_bcs_on_cantileverBeam2d(mesh, lx, ly, i_u, u_expr):
    """
    mesh: a rectangle mesh of lx*ly
    i_u: function_space of the full displacement field
    u_expr: an expression for the displacement load
    
    For a cantilever structure ready for a modelling (e.g. to be given a material type in the future)
    , returns:
        bcs_DR, bcs_DR_dofs, bcs_DR_inhom, bcs_DR_inhom_dofs, [time_depending_expressions]
    """
    assert(mesh.geometric_dimension()==2)
    assert(i_u.num_sub_spaces()==2)
    bcs_DR = []
    bcs_DR_dofs = []
    bcs_DR_inhom = []
    bcs_DR_inhom_dofs = []
    
    def left_edge(x, on_boundary):
        return on_boundary and df.near(x[0], 0., tol)
    bc_left_x, bc_left_x_dofs = boundary_condition(i_u.sub(0), df.Constant(0.0), left_edge) # fix in x-direction at (x[0]=0)
    bcs_DR.append(bc_left_x)
    bcs_DR_dofs.extend(bc_left_x_dofs)
    ## fully clamped
    bc_left_y, bc_left_y_dofs = boundary_condition(i_u.sub(1), df.Constant(0.0), left_edge)
    bcs_DR.append(bc_left_y)
    bcs_DR_dofs.extend(bc_left_y_dofs)
    
    ## load on the right-top node
    def right_top(x, on_boundary):
        return df.near(x[0], lx, tol) and df.near(x[1], ly, tol)
    bc_right, bc_right_dofs = boundary_condition_pointwise(i_u.sub(1), u_expr, right_top)
    bcs_DR_inhom.append(bc_right)
    bcs_DR_inhom_dofs.extend(bc_right_dofs)
    
    return bcs_DR, bcs_DR_dofs, bcs_DR_inhom, bcs_DR_inhom_dofs

def load_and_bcs_on_cantileverBeam3d(mesh, lx, ly, lz, i_u, u_expr):
    """
    Similar to "load_and_bcs_on_cantileverBeam2d" but for 3-D case.
    The cross section of beam is in y-z plane. In a 2-D view of beam, Z-direction is out of plane.
    """
    assert(mesh.geometric_dimension()==3)
    assert(i_u.num_sub_spaces()==3)
    bcs_DR = []
    bcs_DR_dofs = []
    bcs_DR_inhom = []
    bcs_DR_inhom_dofs = []
    
    ## fix x-y DOFs at the left cross section
    def left_section(x, on_boundary):
        return on_boundary and df.near(x[0], 0., tol)
    bc_left_x, bc_left_x_dofs = boundary_condition(i_u.sub(0), df.Constant(0.0), left_section)
    bcs_DR.append(bc_left_x)
    bcs_DR_dofs.extend(bc_left_x_dofs)
    bc_left_y, bc_left_y_dofs = boundary_condition(i_u.sub(1), df.Constant(0.0), left_section)
    bcs_DR.append(bc_left_y)
    bcs_DR_dofs.extend(bc_left_y_dofs)
    ## fix z DOFs at two nodes at the left cross section (the nodes must have the same z-coordinate)
    def left_2nodes(x):
        return (df.near(x[0], 0, tol) and df.near(x[1], 0, tol) and df.near(x[2], 0, tol) ) \
            or (df.near(x[0], 0, tol) and df.near(x[1], ly, tol) and df.near(x[2], 0, tol) )
    bc_left_2nodes, bc_left_2nodes_dofs = boundary_condition_pointwise(i_u.sub(2), df.Constant(0.0), left_2nodes)
    bcs_DR.append(bc_left_2nodes)
    bcs_DR_dofs.extend(bc_left_2nodes_dofs)
    
    ## load on the right-top edge
    def right_top_edge(x):
        return df.near(x[0], lx, tol) and df.near(x[1], ly, tol)
    bc_right, bc_right_dofs = boundary_condition_pointwise(i_u.sub(1), u_expr, right_top_edge)
    bcs_DR_inhom.append(bc_right)
    bcs_DR_inhom_dofs.extend(bc_right_dofs)
    
    return bcs_DR, bcs_DR_dofs, bcs_DR_inhom, bcs_DR_inhom_dofs

def sinusoidal_loading(_vals, intervals_bounds=None, N=1, t0=0.0, T=1.0, _degree=0 \
                               , _plot=True, _res=1000, lab_x='t', lab_y='u', sz=14, _tit='Plot' \
                                   , _save=True, _path='./', _name='plot', _format='.png'):
    if intervals_bounds is None:
        intervals_bounds = np.linspace(t0, T, len(_vals)+1)
    assert len(_vals) == len(intervals_bounds) - 1
    def switcher(t):
        value="not_yet_evaluated"
        if t>=intervals_bounds[0] and t<=intervals_bounds[1]:
            value = _vals[0]
        i = 1
        while value=="not_yet_evaluated" and i<len(_vals):
            if t>intervals_bounds[i] and t<=intervals_bounds[i+1]:
                value = _vals[i]
            i += 1
        if value=="not_yet_evaluated":
            raise RuntimeError('The scalar switcher cannot be evaluated at a point outside of the provided intervals.')
        return value
    nested_expr = df.Expression('sin(N * 2 * p * t / T)', degree=_degree, t=t0, T=T, p=np.pi, N=N)
    expr = TimeVaryingExpressionFromPython(t_callable=switcher, t0=intervals_bounds[0], T=intervals_bounds[-1], _degree=_degree, nested_expr=nested_expr \
                 , _plot=_plot, _res=_res, lab_x=lab_x, lab_y=lab_y, sz=sz, _tit=_tit \
                    , _save=_save, _path=_path, _name=_name, _format=_format)
    return expr

class TimeVaryingExpressionFromPython(df.UserExpression):
    def __init__(self, t_callable, t0=0., T=1.0, _degree=1, nested_expr=df.Expression('val', val=1.0, t=0.0, degree=0) \
                 , _plot=True, _res=1000, lab_x='t', lab_y='u', sz=14, _tit='Plot over time'\
                    , _save=True, _path='./', _name='plot', _format='.png'\
                     , **kwargs):
        super().__init__(degree=_degree, **kwargs)
        self.t_callable = t_callable # a callable of t
        self.t0 = t0 # initial time
        self.Tend = T # end-time
        self.t = 0.0 # time
        self.nested_expr = nested_expr
        if _plot:
            plot_expression_of_t(self, t0=self.t0, t_end=self.Tend, _res=_res, lab_x=lab_x, lab_y=lab_y, sz=sz, _tit=_tit\
                                 , _save=_save, _path=_path, _name=_name, _format=_format)
    def eval(self, values, x):
        self.nested_expr.t = self.t
        values[0] = self.t_callable(self.t) * self.nested_expr(x)
    def value_shape(self):
        return ()

def plot_expression_of_t(expr, t0, t_end, _res=1000, lab_x='t', lab_y='u', sz=14, _tit='Plot'\
                    , _save=True, _path='./', _name='plot', _format='.png'):
    import matplotlib.pyplot as plt
    ts, us = evaluate_expression_of_t(expr, t0=t0, t_end=t_end, _res=_res)
    plt.plot(ts, us)
    plt.title(_tit, fontsize=sz)
    plt.xlabel(lab_x, fontsize=sz)
    plt.ylabel(lab_y, fontsize=sz)
    plt.xticks(fontsize=sz)
    plt.yticks(fontsize=sz)
    if _save:
        import os
        if not os.path.exists(_path):
            os.makedirs(_path)
        plt.savefig(_path + _name + _format)
    plt.show()
    
def evaluate_expression_of_t(expr, t0, t_end, _res=1000):
    ts = np.linspace(t0, t_end, _res + 1)
    mesh = df.UnitIntervalMesh(1)
    e = df.FiniteElement('R', mesh.ufl_cell(), degree=0)
    V = df.FunctionSpace(mesh, e)
    v = df.TestFunction(V)
    ff = expr * v * df.dx
    us = []
    for tt in ts:
        expr.t = tt
        us.append(sum(df.assemble(ff).get_local()))
    return ts, us

class PostProcess:
    def __init__(self, fen, _name='', out_path=None, reaction_dofs=None, log_residual_vector=False, write_files=True):
        self.fen = fen
        self.reaction_dofs = reaction_dofs # for which, the reaction force (residual) will be calculated
        self.log_residual_vector = log_residual_vector
        
        self.ts = [] # list of times
        self.checkpoints = [] # will be assigned as soon as a time-stepper incremental solution is executed.
        
        self.reaction_forces = [] # Every entry is for a certain time ("t") and is a list of values over the given reaction_dofs
        
        if out_path is None:
            out_path = './'
        make_path(out_path)
        self.full_name = out_path + _name
        
        self.write_files = write_files
        self.remove_files()
        
        self.u_checked_xdmf = df.XDMFFile(self.full_name + '_u_checked.xdmf') # This stores the full Function with function-space data.
        
        self.reaction_forces_checked = h5py.File(self.full_name + '_reaction_forces.h5', 'w') ## to save reaction forces to HTF5 file
        
        self.last_checked_id = -1 # the first checked ID will be then "0" ---> zero-base IDs
        
        if self.write_files:
            self.u_xdmf = df.XDMFFile(self.full_name + '_u.xdmf') # This is just for visualization
        
    def __call__(self, t):
        self.ts.append(t)
        
        F=self.fen.get_F_and_u()[0]
        res = list(df.assemble(F).get_local())
        if self.reaction_dofs is None:
            reaction_force = None
        else:
            reaction_force = [res[i] for i in self.reaction_dofs]
        
        u_plot = self.fen.get_uu()
        for tt in self.checkpoints:
            if abs(t - tt) < 1e-9:
                self.u_checked_xdmf.write_checkpoint(u_plot, 'u', t, append=True)
                if reaction_force is not None:
                    self.last_checked_id += 1
                    self.reaction_forces_checked.create_dataset('f_' + str(self.last_checked_id), data=reaction_force)
                break
        
        if reaction_force is not None:
            self.reaction_forces.append(reaction_force)
        
        if self.write_files:
            self.u_xdmf.write(u_plot, t)
    
    def plot_reaction_forces(self, tit, dof='sum', full_file_name=None, factor=1, marker='.', sz=14):
        fig1 = plt.figure()
        if dof=='sum':
            if type(self.reaction_forces[0])==list or type(self.reaction_forces[0])==np.ndarray:
                f_dof = [factor * sum(f) for f in self.reaction_forces]
            else:
                f_dof = [factor * f for f in self.reaction_forces]
        else:
            f_dof = [factor * f[dof] for f in self.reaction_forces]
        plt.plot(self.ts, f_dof, marker=marker)
        plt.title(tit, fontsize=sz)
        plt.xlabel('t', fontsize=sz)
        plt.ylabel('f', fontsize=sz)
        plt.xticks(fontsize=sz)
        plt.yticks(fontsize=sz)
        
        if full_file_name is not None:
            plt.savefig(full_file_name)
        plt.show()
        
    def close_files(self):
        self.u_checked_xdmf.close()
        self.reaction_forces_checked.close()
        if self.write_files:
            self.u_xdmf.close()
    
    def eval_checked_u(self, points):
        v = self.fen.get_iu(_collapse=True)
        u_read = df.Function(v)
        vals = []
        num_checkpoints = len(self.checkpoints)
        for ts in range(num_checkpoints):
            self.u_checked_xdmf.read_checkpoint(u_read, 'u', ts)
            u_ts = [u_read(p) for p in points]
            vals.append(np.array(u_ts))
        return vals
    
    def eval_checked_reaction_forces(self):
        hf = h5py.File(self.full_name  + '_reaction_forces.h5', 'r')
        fs = []
        for ts in range(self.last_checked_id + 1):
            fs.append(list(hf.get('f_' + str(ts))))
        return fs
    
    def remove_files(self):
        import os
        if os.path.exists(self.full_name + '_u_checked.xdmf'):
            os.remove(self.full_name + '_u_checked.xdmf')
            os.remove(self.full_name + '_u_checked.h5')
        if os.path.exists(self.full_name + '_reaction_forces.h5'):
            os.remove(self.full_name + '_reaction_forces.h5')
        if os.path.exists(self.full_name + '_u.xdmf') and self.write_files:
            os.remove(self.full_name + '_u.xdmf')
            os.remove(self.full_name + '_u.h5')
            
class PostProcessPlastic(PostProcess):
    def __init__(self, fen, _name='', out_path=None, reaction_dofs=None, log_residual_vector=False, write_files=True):
        super().__init__(fen, _name, out_path, reaction_dofs, log_residual_vector, write_files)
        if self.write_files:
            self.remove_files_2()
            self.sig_xdmf = df.XDMFFile(self.full_name + '_sigma.xdmf')
            self.eps_p_xdmf = df.XDMFFile(self.full_name + '_eps_p.xdmf')
            self.K_xdmf = df.XDMFFile(self.full_name + '_Kappa.xdmf')
            
            elem_dg0_k = df.FiniteElement("DG", self.fen.mesh.ufl_cell(), degree=0)
            elem_dg0_ss = df.VectorElement("DG", self.fen.mesh.ufl_cell(), degree=0, dim=self.fen.mat.ss_dim)
            self.i_k = df.FunctionSpace(self.fen.mesh, elem_dg0_k)
            self.i_ss = df.FunctionSpace(self.fen.mesh, elem_dg0_ss)
            
            self.sig = df.Function(self.i_ss, name='Stress')
            self.eps_p = df.Function(self.i_ss, name='Cumulated plastic strain')
            self.K = df.Function(self.i_k, name='Cumulated Kappa')

    def __call__(self, t):
        super().__call__(t)
        if self.write_files:
            ### project from quadrature space to DG-0 space
            df.project(v=self.fen.sig, V=self.i_ss, function=self.sig)
            df.project(v=self.fen.eps_p, V=self.i_ss, function=self.eps_p)
            df.project(v=self.fen.kappa, V=self.i_k, function=self.K)
            ### write projected values to xdmf-files
            self.K_xdmf.write(self.K, t)
            self.sig_xdmf.write(self.sig, t)
            self.eps_p_xdmf.write(self.eps_p, t)
    
    def close_files(self):
        super().close_files()
        if self.write_files:
            self.sig_xdmf.close()
            self.eps_p_xdmf.close()
            self.K_xdmf.close()
        
    def remove_files_2(self): # different than remove_files() of the supperclass and NOT extension of that.
        import os
        if os.path.exists(self.full_name + '_sigma.xdmf'):
            os.remove(self.full_name + '_sigma.xdmf')
            os.remove(self.full_name + '_sigma.h5')
        if os.path.exists(self.full_name + '_eps_p.xdmf'):
            os.remove(self.full_name + '_eps_p.xdmf')
            os.remove(self.full_name + '_eps_p.h5')
        if os.path.exists(self.full_name + '_Kappa.xdmf'):
            os.remove(self.full_name + '_Kappa.xdmf')
            os.remove(self.full_name + '_Kappa.h5')