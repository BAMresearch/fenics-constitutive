"""
Helper classes and functions
============================

Suppress warnings
-----------------

The whole quadrature space is half deprecated, half not. We roll with it 
and just ignore the warnings.
"""

import numpy as np

import dolfin as df


def setup(module):
    import warnings
    from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning

    module.parameters["form_compiler"]["representation"] = "quadrature"
    warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)

    try:
        from fenics_helpers import boundary
        from fenics_helpers.timestepping import TimeStepper
    except Exception as e:
        print("Install fenics_helpers via (e.g.)")
        print("   pip3 install git+https://github.com/BAMResearch/fenics_helpers")
        raise (e)


setup(df)


"""
Load-displacement curve
-----------------------

For a given state (e.g. a time step), load-displacement curve connects the 
displacements of certain DOFs with their reaction forces (*load*). Especially
for strain-softening materials, this curve can indicate numerical issues
like snap-backs.

From ``dolfin.DirichletBC.get_boundary_values()`` , we directly extract the 
``.values()`` as the current displacements and the ``.keys()`` as the DOF 
numbers. The latter ones are used to extract the reaction (out-of-balance) 
forces from a given force vector ``R``.

Special care is taken to make this class work in parallel.
"""


class LoadDisplacementCurve:
    def __init__(self, bc):
        """
        bc:
            dolfin.DirichletBC 
        """
        self.comm = df.MPI.comm_world
        self.bc = bc

        self.dofs = list(self.bc.get_boundary_values().keys())
        self.n_dofs = df.MPI.sum(self.comm, len(self.dofs))

        self.load = []
        self.disp = []
        self.ts = []
        self.plot = None

        self.is_root = df.MPI.rank(self.comm) == 0

    def __call__(self, t, R):
        """
        t:
            global time
        R:
            residual, out of balance forces
        """
        # A dof > R.local_size() is (I GUESS?!?!) a ghost node and its
        # contribution to R is accounted for on the owning process. So it
        # is (I GUESS?!) safe to ignore it.
        self.dofs = [d for d in self.dofs if d < R.local_size()]

        load_local = np.sum(R[self.dofs])
        load = df.MPI.sum(self.comm, load_local)

        disp_local = np.sum(list(self.bc.get_boundary_values().values()))
        disp = df.MPI.sum(self.comm, disp_local) / self.n_dofs

        self.load.append(load)
        self.disp.append(disp)
        self.ts.append(t)
        if self.plot and self.is_root:
            self.plot(disp, load)

    def show(self, fmt="-rx"):
        if self.is_root:
            try:
                from fenics_helpers.plotting import AdaptivePlot

                self.plot = AdaptivePlot(fmt)
            except ImportError:
                print("Skip LD.show() because matplotlib.pyplot cannot be imported.")

    def keep(self):
        if self.is_root:
            self.plot.keep()


"""
Local projector
---------------

Projecting an expression ``expr(u)`` into a function space ``V`` is done by
solving the variational problem

.. math::
    \int_\Omega uv \ \mathrm dx = \int_\omega \text{expr} \ v \ \mathrm dx

for all test functions $v \in V$.

In our case, $V$ is a quadrature function space and can significantly speed
up this solution by using the ``dolfin.LocalSolver`` that can additionaly
be prefactorized to speedup subsequent projections.
"""


class LocalProjector:
    def __init__(self, expr, V, dxm):
        """
        expr:
            expression to project
        V:
            quadrature function space
        dxm:
            dolfin.Measure("dx") that matches V
        """
        dv = df.TrialFunction(V)
        v_ = df.TestFunction(V)
        a_proj = df.inner(dv, v_) * dxm
        b_proj = df.inner(expr, v_) * dxm
        self.solver = df.LocalSolver(a_proj, b_proj)
        self.solver.factorize()

    def __call__(self, u):
        """
        u:
            function that is filled with the solution of the projection
        """
        self.solver.solve_local_rhs(u)


"""
Setting values for the quadrature space
---------------------------------------

* The combination of ``.zero`` and ``.add_local`` is faster than using
  ``.set_local`` directly, as ``.set_local`` copies everything in a `C++ vector <https://bitbucket.org/fenics-project/dolfin/src/946dbd3e268dc20c64778eb5b734941ca5c343e5/python/src/la.cpp#lines-576>`__ first.
* ``.apply("insert")`` takes care of the ghost value communication in a 
  parallel run.

"""


def set_q(q, values):
    """
    q:
        quadrature function space
    values:
        entries for `q`
    """
    v = q.vector()
    v.zero()
    v.add_local(values.flatten())
    v.apply("insert")


def spaces(mesh, deg_q, qdim):
    cell = mesh.ufl_cell()
    q = "Quadrature"
    QF = df.FiniteElement(q, cell, deg_q, quad_scheme="default")
    QV = df.VectorElement(q, cell, deg_q, quad_scheme="default", dim=qdim)
    QT = df.TensorElement(q, cell, deg_q, quad_scheme="default", shape=(qdim, qdim))
    return [df.FunctionSpace(mesh, Q) for Q in [QF, QV, QT]]


"""
.. _target-orient:

Reading orientations from Abaqus input file(s)
----------------------------------------------

+ Create an orientation class (for example class `ori`) ``ori = OrientationFromAbaqus()``.

+ Read the Abaqus input file(s) in the created class ``ori.read_orientations_from("AbaqusInputFile.inp")``.

+ If the Abaqus options \*ORIENTATION and \*SOLID SECTION are in different input files, then the call 
  must be performed for each of these files ``ori.read_orientations_from("AbaqusInputFileContainingOrientations.inp")`` and ``ori.read_orientations_from("AbaqusInputFileContainingSolidSections.inp")``.

+ Call ``ori.isComplete()`` to ensure that the computation from \*ORIANTATION to Euler 
  angles has been internally done within the `ori` class. The computation cannot be carried out if the 
  input data are incomplete.

+ Access the Euler angles via the dictionary ``ori.orientation_data_Euler``, which provides 
  `np.array([phi, theta, rho])` in degrees for a given solid section name.

"""

import os.path
import sys

class OrientationFromAbaqus:
    def __init__(self):
        self.orientation_data_Abaqus = {}
        self.orientation_data_Euler   = {}
        self.solid_section_data      = {}

    def read_orientations_from(self, filename):
        # Reads Abaqus orientations from an inp file, which contains
        # *ORIENT and/or *SOLID SECTION
        # if *ORIENT and *SOLID SECTION are in different files,
        # the function must be called sequentially
        if not os.path.isfile(filename):
            print("**ERROR: file with orientations ", filename, " was not found.")
            raise Exception()

        with open(filename, "r") as f:
            orientation_data   = {}
            solid_saction_data = {}
            line = f.readline()
            while True:
                if not line:
                    break
                
                if line.startswith("**"):
                    line = f.readline()
                    continue
                    
                keyword = line.partition(",")[0].strip().replace("*", "").upper()
                if keyword == "ORIENTATION":
                    
                    param_map = self.evaluate_argument_line(line)
                    self.orientation_data_Abaqus[param_map['NAME']] = self.read_rectangular_values(f)

                    # go to next orientation
                    line = f.readline()
                    continue
                if keyword == "SOLID SECTION":

                    param_map = self.evaluate_argument_line(line)
                    self.solid_section_data[param_map['ELSET']] =[param_map['MATERIAL'],param_map['ORIENTATION']]
                    
                    # go to next solid section
                    line = f.readline()
                    continue
                else:
                    line = f.readline()

            # if orientations have been read, convert them to Euler angles
            if bool(self.orientation_data_Abaqus) and not bool(self.orientation_data_Euler):
                for key, OriAbaqus in self.orientation_data_Abaqus.items():
                    EulerAngles = self.convert2Euler(OriAbaqus[0])
                    self.orientation_data_Euler[key] = EulerAngles
        
    def isComplete(self):
        if bool(self.solid_section_data) and bool(self.orientation_data_Euler):
            return True
        
    def evaluate_argument_line(self, line):
        # reads line, which starts by *keyword, like *Orientation, *Solid Section etc.
        # returns a dict., like
        # {SYSTEM: RECTANGULAR, NAME: ORIENT1}
        # for line: *ORIENTATION, SYSTEM=RECTANGULAR, NAME=ORIENT1

        words = line.split(",")
        param_map = {}
        for word in words:
            if "=" in word:
                sword = word.split("=")
                
                if len(sword) != 2:
                    print("**ERROR: wrong input in ",line)
                    raise Exception()

                key = sword[0].strip().upper()
                value = sword[1].strip()
                param_map[key] = value
                
        keyword = words[0].strip().replace("*", "").upper()
        if keyword == 'ORIENTATION':
            if param_map['SYSTEM'] != 'RECTANGULAR':
                print("**ERROR: only RECTANGULAR system is implemented.")
                raise Exception()
        if keyword == 'SOLID SECTION':
            if 'ELSET' not in list(param_map.keys()):
                print("**ERROR: ELSET missing in ",line)
                raise Exception()
            if 'MATERIAL' not in list(param_map.keys()):
                print("**ERROR: MATERIAL missing in ",line)
                raise Exception()
            if 'ORIENTATION' not in list(param_map.keys()):
                print("**ERROR: ORIENTATION missing in ",line)
                raise Exception()
            
        return param_map
    
    def read_rectangular_values(self, f):
        values = []
        while True:
            line = f.readline()
            if not line or line.startswith("*"):
                break
            if line.strip() == "":
                continue

            words = line.strip().split(",")
        
            if len(words) < 6:
                print("**ERROR: wrong input in ",line)
                raise Exception()

            values.append([float(x) for x in words[:6]])
            # only one line must be read
            break
        return np.array(values, dtype=float)

    def convert2Euler(self, orientationAbaqus):
        # returns np.array [phi, theta, rho]
        # notations Sp := sin(phi), Ct := cos(theta) etc.
        
        # the system solved is:
        # a1 =  CrCp - CtSrSp;  a2 = CtCpSr + CrSp; a3 = StSr
        # b1 = -CpSr - CtCrSp;  b2 = CtCrCp - SrSp; b3 = CrCt
        
        # some equalities:
        # TANr = a3/b3
        #
        # TANp = b3/a3 (b2 - Ct a1)/(b2 Ct - a1) or TANp = - (b1 + a1 TANr)/(a2 TANr + b2)
        # obtained by (a1 Sr + b1 C)r / (a2 Sr + b2 Cr)
        #
        # Ct = - (a2 TANr + b2)/(b1 TANr - a1)
        # obtained by (a1 Ct - b2) / (b1 Ct + a2)
        
        a1 = orientationAbaqus[0]
        a2 = orientationAbaqus[1]
        a3 = orientationAbaqus[2]
        b1 = orientationAbaqus[3]
        b2 = orientationAbaqus[4]
        b3 = orientationAbaqus[5]

        TANr = a3/b3
        rho = np.arctan2(a3,b3)
   
        Ct = -(b2 + a2*TANr)/(b1*TANr - a1)
        St = np.sqrt(a3*a3 + b3*b3)
        theta = np.arccos(Ct)
    
        TANp = -(b1 + a1*TANr)/(a2*TANr + b2)
        phi = np.arctan(TANp)
   
        # the convertion is not unique
        # the solutions must be checked with regards to +/-pi
        convertion = False
        for angle in [phi, phi + np.pi, phi - np.pi]:
            A1 =  np.cos(rho)*np.cos(angle) - np.cos(theta)*np.sin(rho)*np.sin(angle)
            A2 =  np.cos(rho)*np.sin(angle) + np.cos(theta)*np.cos(angle)*np.sin(rho)
            B1 = -np.cos(angle)*np.sin(rho) - np.cos(theta)*np.cos(rho)*np.sin(angle)
            B2 = -np.sin(rho)*np.sin(angle) + np.cos(theta)*np.cos(angle)*np.cos(rho)
            # a3 and b3 are correct and do not depend on phi
            if np.allclose([a1,a2,b1,b2],[A1,A2,B1,B2]):
                phi = angle
                convertion = True
                break

        if not convertion:
            print("**ERROR: convertion of ORIENTATION to Euler not found.")
            raise Exception()

        return np.array([phi, theta, rho])*180/np.pi
