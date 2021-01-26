import unittest
from constitutive import *


class Parameters:
    def __init__(self, constraint):
        self.constraint = constraint.lower()

        # interpolation degree of displacement field
        self.deg_d = 2

        # quadrature degree
        self.deg_q = 2

        self.E = 20000
        self.nu = 0.2


class Base:
    def __init__(self, mat):
        self.E, self.nu = mat.E, mat.nu
        if mat.constraint == "uniaxial_strain":
            self.eps = self._eps_1d
            self.C = self._uniaxial_strain()
            self.g_dim = 1
        elif mat.constraint == "uniaxial_stress":
            self.eps = self._eps_1d
            self.C = self._uniaxial_stress()
            self.g_dim = 1
        elif mat.constraint == "plane_strain":
            self.eps = self._eps_2d
            self.C = self._plane_strain()
            self.g_dim = 2
        elif mat.constraint == "plane_stress":
            self.eps = self._eps_2d
            self.C = self._plane_stress()
            self.g_dim = 2
        elif mat.constraint == "3d" or mat.constraint == "full":
            self.eps = self._eps_3d
            self.C = self._full()
            self.g_dim = 3
        else:
            raise RuntimeError(
                "The constraint must be uniaxial_stress",
                "uniaxial_strain, plane_stress, plane_strain or full/3d.",
            )

        self.q_dim = self.C.shape[0]

    def _eps_1d(self, u):
        e = df.sym(df.grad(u))
        return df.as_vector([e[0, 0]])

    def _eps_2d(self, u):
        e = df.sym(df.grad(u))
        return df.as_vector([e[0, 0], e[1, 1], 2 * e[0, 1]])

    def _eps_3d(self, u):
        e = df.sym(df.grad(u))
        return df.as_vector(
            [e[0, 0], e[1, 1], e[2, 2], 2 * e[1, 2], 2 * e[0, 2], 2 * e[0, 1]]
        )

    def _uniaxial_strain(self):
        return np.atleast_2d(self.E)

    def _uniaxial_stress(self):
        return np.atleast_2d(self.E)

    def _plane_strain(self):
        l = self.E * self.nu / (1 + self.nu) / (1 - 2 * self.nu)
        m = self.E / (2.0 * (1 + self.nu))
        return np.array([[2 * m + l, l, 0], [l, 2 * m + l, 0], [0, 0, m]])

    def _plane_stress(self):
        C11 = self.E / (1 - self.nu ** 2)
        C12 = self.nu * C11
        C33 = (1 - self.nu) / 2 * C11
        return np.array([[C11, C12, 0], [C12, C11, 0], [0, 0, C33]])

    def _full(self):
        l = self.E * self.nu / (1 + self.nu) / (1 - 2 * self.nu)
        m = self.E / (2.0 * (1 + self.nu))
        matrix = np.diag((2 * m, 2 * m, 2 * m, m, m, m))
        matrix[:3, :3] += l
        return matrix


class MechanicsProblem(df.NonlinearProblem):
    def __init__(self, mesh, mat):
        df.NonlinearProblem.__init__(self)
        self.mat = mat
        self.base = Base(self.mat)
        if mesh.geometric_dimension() != self.base.g_dim:
            raise RuntimeError(
                f"The geometric dimension of the mesh does not match the {mat.constraint} constraint."
            )

        metadata = {"quadrature_degree": mat.deg_q, "quadrature_scheme": "default"}
        dxm = df.dx(metadata=metadata)

        # solution field
        self.V = df.VectorFunctionSpace(mesh, "CG", degree=mat.deg_d)
        self.d = df.Function(self.V, name="displacement field")

        # generic quadrature function spaces
        VQF, VQV, VQT = h.spaces(mesh, mat.deg_q, self.base.q_dim)

        # quadrature function
        self.q_sigma = df.Function(VQV, name="current stresses")
        self.q_eps = df.Function(VQV, name="current strains")
        self.q_dsigma_deps = df.Function(VQT, name="stress-strain tangent")

        dd, d_ = df.TrialFunction(self.V), df.TestFunction(self.V)

        eps = self.base.eps
        self.R = df.inner(eps(d_), self.q_sigma) * dxm
        self.dR = df.inner(eps(dd), self.q_dsigma_deps * eps(d_)) * dxm

        self.calculate_eps = h.LocalProjector(eps(self.d), VQV, dxm)

        self.assembler = None

    @property
    def Vd(self):
        """
        Return the function space for the displacements. This distinction is
        required for mixed problems where V != Vd.
        """
        return self.V
    
    @property
    def u(self):
        """
        Return the full (for all dofs) solution field. 
        """
        return self.d

    def eps_plane_strain(self, u):
        e = df.sym(df.grad(u))
        return df.as_vector([e[0, 0], e[1, 1], 2 * e[0, 1]])

    def evaluate_material(self):
        # project the strain and the nonlocal equivalent strains onto
        # their quadrature spaces and ...
        self.calculate_eps(self.q_eps)
        eps = self.q_eps.vector().get_local().reshape(-1, self.base.q_dim)

        # ... "manually" evaluate_material the material ...
        C = self.base.C
        sigma = eps @ C
        dsigma = np.tile(C.flatten(), len(eps))

        # ... and write the calculated values into their quadrature spaces.
        h.set_q(self.q_sigma, sigma)
        h.set_q(self.q_dsigma_deps, dsigma)

    def update(self):
        pass

    def set_bcs(self, bcs):
        # Only now (with the bcs) can we initialize the assembler
        self.assembler = df.SystemAssembler(self.dR, self.R, bcs)

    def F(self, b, x):
        if not self.assembler:
            raise RuntimeError("You need to `.set_bcs(bcs)` before the solve!")
        self.evaluate_material()
        self.assembler.assemble(b, x)

    def J(self, A, x):
        self.assembler.assemble(A)


class TestUniaxial(unittest.TestCase):
    def test_mismatch(self):
        mat = Parameters("plane_strain")
        mesh = df.UnitIntervalMesh(10)
        self.assertRaises(Exception, MechanicsProblem, mesh, mat)

    def test_1d(self):
        mat = Parameters("uniaxial_strain")
        mesh = df.UnitIntervalMesh(10)

        u_bc = 42.0
        problem = MechanicsProblem(mesh, mat)
        bcs = []
        bcs.append(df.DirichletBC(problem.Vd, [0], bc.plane_at(0)))
        bcs.append(df.DirichletBC(problem.Vd, [u_bc], bc.plane_at(1)))
        problem.set_bcs(bcs)

        solver = df.NewtonSolver()
        solver.solve(problem, problem.u.vector())

        xs = np.linspace(0, 1, 5)
        for x in xs:
            u_fem = problem.u((x))
            u_correct = x * u_bc
            self.assertAlmostEqual(u_fem, u_correct)

    def test_plane_strain(self):
        mat = Parameters("plane_strain")
        mesh = df.UnitSquareMesh(10, 10)

        u_bc = 42.0
        problem = MechanicsProblem(mesh, mat)
        bcs = []
        bcs.append(df.DirichletBC(problem.Vd.sub(0), 0, bc.plane_at(0)))
        bcs.append(df.DirichletBC(problem.Vd.sub(0), u_bc, bc.plane_at(1)))
        bcs.append(df.DirichletBC(problem.Vd.sub(1), 0, bc.plane_at(0, "y")))
        problem.set_bcs(bcs)

        solver = df.NewtonSolver()
        solver.solve(problem, problem.u.vector())

        xs = np.linspace(0, 1, 5)
        for x in xs:
            for y in xs:
                u_fem = problem.u((x, y))
                u_correct = (x * u_bc, -y * u_bc * (mat.nu) / (1 - mat.nu))
                self.assertAlmostEqual(u_fem[0], u_correct[0])
                self.assertAlmostEqual(u_fem[1], u_correct[1])

    def test_3d(self):
        mat = Parameters("3d")
        mesh = df.UnitCubeMesh(5, 5, 5)

        u_bc = 42.0
        problem = MechanicsProblem(mesh, mat)
        bcs = []
        bcs.append(df.DirichletBC(problem.Vd.sub(0), 0, bc.plane_at(0)))
        bcs.append(df.DirichletBC(problem.Vd.sub(0), u_bc, bc.plane_at(1)))
        bcs.append(df.DirichletBC(problem.Vd.sub(1), 0, bc.plane_at(0, "y")))
        bcs.append(df.DirichletBC(problem.Vd.sub(2), 0, bc.plane_at(0, "z")))
        problem.set_bcs(bcs)

        solver = df.NewtonSolver()
        solver.solve(problem, problem.u.vector())

        xs = np.linspace(0, 1, 5)
        for x in xs:
            for y in xs:
                for z in xs:
                    u_fem = problem.u((x, y, z))
                    # print(u_fem)
                    u_correct = (x * u_bc, -y * u_bc * mat.nu, -z * u_bc * mat.nu)
                    self.assertAlmostEqual(u_fem[0], u_correct[0])
                    self.assertAlmostEqual(u_fem[1], u_correct[1])
                    self.assertAlmostEqual(u_fem[2], u_correct[2])


if __name__ == "__main__":
    unittest.main()
