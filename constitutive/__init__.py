from .cpp import *

try:
    import dolfinx as dfx
except ModuleNotFoundError:
    import dolfin as df

    import warnings

    from ffc.quadrature.deprecation import \
        QuadratureRepresentationDeprecationWarning

    warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)
    df.parameters["form_compiler"]["representation"] = "quadrature"
    from .explicit_dynamics import *
    from .helper import *
    from .mechanics_problem import *

    try:
       from fenics_helpers import boundary as bc
       from fenics_helpers.timestepping import TimeStepper
    except Exception as e:
       print("Install fenics_helpers via (e.g.)")
       print("   pip3 install git+https://github.com/BAMResearch/fenics_helpers")
       raise (e)
