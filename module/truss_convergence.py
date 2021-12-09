"""
truss under its own dead weight

solve
    EA u''(x) = - ρgA on Ω

analytical solution
    u(x) = ρgl (x - x ** 2 / 2 / l) / E
    with Ω = (0, l)
"""
from dataclasses import dataclass
import dolfin as df

@dataclass
class TrussSolution:
    E: float
    L: float
    rho: float
    A: float
    g: float

    def displacement(self, x):
        return self.rho * self.g * self.L * (x[0] - x[0] ** 2 / 2 / self.L) / self.E


class DisplacementSolution(df.UserExpression):
    def __init__(self, solution, **kwargs):
        super().__init__(**kwargs)
        self.solution = solution

    def eval(self, value, x):
        u = self.solution.displacement(x)
        value[0] = u

    def value_shape(self):
        return ()


def generate_mesh(L):
    """function to be provided by the user to generate mesh

    Parameters
    ----------
    depends on specific case

    Returns
    -------
    mesh: 
        the mesh 
    """
    return df.IntervalMesh(1, 0., L)

def solve_truss_problem(mesh, params, degree):
    """solve the model on given mesh"""
    V = df.FunctionSpace(mesh, "Lagrange", degree)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    E = params["E"]
    g = params["g"]
    A = params["A"]
    rho = params["rho"]
    f = df.Constant(rho * g * A)
    a = E * df.inner(df.grad(u), df.grad(v)) * df.dx
    L = f * v * df.dx

    def left(x, on_boundary):
        return x[0] < df.DOLFIN_EPS and on_boundary

    bcs = df.DirichletBC(V, df.Constant(0.0), left)
    solution = df.Function(V)
    df.solve(a == L, solution, bcs)
    return solution


def get_exact_solution(parameters):
    """some function to provide reference solution"""
    exact = TrussSolution(**parameters)
    u = DisplacementSolution(exact, degree=2)
    return u


if __name__ == "__main__":
    parameters = {
        "L": 42.0,
        "E": 10.0,
        "g": 10.0,
        "A": 1.0,
        "rho": 1.0,
        }
    n_refinements = 0
    u = get_exact_solution(parameters)
    mesh = generate_mesh(parameters["L"])
    print(mesh.num_cells())
    while True:
        u_fem = solve_truss_problem(mesh, parameters, degree=1)
        err = df.errornorm(u, u_fem, norm_type="l2", mesh=mesh)
        if err < 1.e-4:
            break

        mesh = df.refine(mesh)
        n_refinements += 1

    print(f"Finally converged. Please use {n_refinements=}.")

