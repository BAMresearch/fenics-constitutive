# Gradient Damage
# ===============
#
# Some text here
#
# .. math::
#
#    (a + b)^2 = a^2 + 2ab + b^2
#
#    (a - b)^2 = a^2 - 2ab + b^2



from dolfin import *
mesh = UnitSquareMesh(40,40)

def my_pretty_function(some_arg):
    """
    Documentation of the function
    """
    return "hallo"


# Formulation
# -----------
# Another important aspect with :math:`\frac{1}{2}` bla bli blub.
#
# .. math::
#
#    (a + b)^2 = a^2 + 2ab + b^2 \\
#    (a - b)^2 = a^2 - 2ab + b^2


class LocalProjector:
    def __init__(self):
        pass

    def new_method(self):
        pass
