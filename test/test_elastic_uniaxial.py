import unittest
from constitutive import *


class Parameters:
    def __init__(self, constraint):
        self.constraint = constraint

        # interpolation degree of displacement field
        self.deg_d = 2

        # quadrature degree
        self.deg_q = 2

        self.E = 20000
        self.nu = 0.2


class MechanicsProblem(df.NonlinearProblem):
    def __init__(self, mesh, prm):
        df.NonlinearProblem.__init__(self)

        self.mesh = mesh
        self.prm = prm

        self.law = LinearElastic(prm.E, prm.nu, prm.constraint)

        self.base = Base(self.law)

        if mesh.geometric_dimension() != g_dim(prm.constraint):
            raise RuntimeError(
                f"The geometric dimension of the mesh does not match the {prm.constraint} constraint."
            )

        metadata = {"quadrature_degree": prm.deg_q, "quadrature_scheme": "default"}
        dxm = df.dx(metadata=metadata)

        # solution field
        self.V = df.VectorFunctionSpace(mesh, "CG", degree=prm.deg_d)
        self.d = df.Function(self.V, name="displacement field")

        # generic quadrature function spaces
        VQF, VQV, VQT = h.spaces(mesh, prm.deg_q, q_dim(prm.constraint))

        # quadrature function
        self.q_sigma = df.Function(VQV, name="current stresses")
        self.q_eps = df.Function(VQV, name="current strains")
        self.q_dsigma_deps = df.Function(VQT, name="stress-strain tangent")

        dd, d_ = df.TrialFunction(self.V), df.TestFunction(self.V)

        eps = self.eps
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

    def eps(self, u):
        e = df.sym(df.grad(u))
        dim = self.mesh.geometric_dimension()
        if dim == 1:
            return df.as_vector([e[0, 0]])

        if dim == 2:
            return df.as_vector([e[0, 0], e[1, 1], 2 * e[0, 1]])

        if dim == 3:
            return df.as_vector(
                [e[0, 0], e[1, 1], e[2, 2], 2 * e[1, 2], 2 * e[0, 2], 2 * e[0, 1]]
            )

    def evaluate_material(self):
        # project the strain and the nonlocal equivalent strains onto
        # their quadrature spaces and ...
        self.calculate_eps(self.q_eps)

        self.base.evaluate(self.q_eps.vector().get_local())

        eps = self.q_eps.vector().get_local().reshape(-1, q_dim(self.prm.constraint))

        # ... "manually" evaluate_material the material ...
        s, C = self.law.evaluate((0, 0, 0))
        sigma = eps @ C
        dsigma = np.tile(C.flatten(), len(eps))

        # ... and write the calculated values into their quadrature spaces.
        h.set_q(self.q_sigma, self.base.stress)
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
        prm = Parameters(Constraint.PLANE_STRAIN)
        mesh = df.UnitIntervalMesh(10)
        self.assertRaises(Exception, MechanicsProblem, mesh, prm)

    def test_1d(self):
        prm = Parameters(Constraint.UNIAXIAL_STRAIN)
        mesh = df.UnitIntervalMesh(10)

        u_bc = 42.0
        problem = MechanicsProblem(mesh, prm)
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
        prm = Parameters(Constraint.PLANE_STRAIN)
        mesh = df.UnitSquareMesh(10, 10)

        u_bc = 42.0
        problem = MechanicsProblem(mesh, prm)
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
                u_correct = (x * u_bc, -y * u_bc * (prm.nu) / (1 - prm.nu))
                self.assertAlmostEqual(u_fem[0], u_correct[0])
                self.assertAlmostEqual(u_fem[1], u_correct[1])

    def test_3d(self):
        prm = Parameters(Constraint.FULL)
        mesh = df.UnitCubeMesh(5, 5, 5)

        u_bc = 42.0
        problem = MechanicsProblem(mesh, prm)
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
                    u_correct = (x * u_bc, -y * u_bc * prm.nu, -z * u_bc * prm.nu)
                    self.assertAlmostEqual(u_fem[0], u_correct[0])
                    self.assertAlmostEqual(u_fem[1], u_correct[1])
                    self.assertAlmostEqual(u_fem[2], u_correct[2])


if __name__ == "__main__":
    unittest.main()
