import unittest
import dolfin as df
import numpy as np
from fenics_helpers import boundary
import constitutive as c

class TestUniaxial(unittest.TestCase):
    def test_mismatch(self):
        prm = c.Parameters(c.Constraint.PLANE_STRAIN)
        mesh = df.UnitIntervalMesh(10)
        self.assertRaises(Exception, c.MechanicsProblem, mesh, prm)

    def test_1d(self):
        prm = c.Parameters(c.Constraint.UNIAXIAL_STRAIN)
        mesh = df.UnitIntervalMesh(10)

        u_bc = 42.0
        problem = c.MechanicsProblem(mesh, prm)
        bcs = []
        bcs.append(df.DirichletBC(problem.Vd, [0], boundary.plane_at(0)))
        bcs.append(df.DirichletBC(problem.Vd, [u_bc], boundary.plane_at(1)))
        problem.set_bcs(bcs)

        u = problem.solve()

        xs = np.linspace(0, 1, 5)
        for x in xs:
            u_fem = u((x))
            u_correct = x * u_bc
            self.assertAlmostEqual(u_fem, u_correct)

    def test_plane_strain(self):
        prm = c.Parameters(c.Constraint.PLANE_STRAIN)
        mesh = df.UnitSquareMesh(10, 10)

        u_bc = 42.0
        problem = c.MechanicsProblem(mesh, prm)
        bcs = []
        bcs.append(df.DirichletBC(problem.Vd.sub(0), 0, boundary.plane_at(0)))
        bcs.append(df.DirichletBC(problem.Vd.sub(0), u_bc, boundary.plane_at(1)))
        bcs.append(df.DirichletBC(problem.Vd.sub(1), 0, boundary.plane_at(0, "y")))
        problem.set_bcs(bcs)

        u = problem.solve()

        xs = np.linspace(0, 1, 5)
        for x in xs:
            for y in xs:
                u_fem = u((x, y))
                u_correct = (x * u_bc, -y * u_bc * (prm.nu) / (1 - prm.nu))
                self.assertAlmostEqual(u_fem[0], u_correct[0])
                self.assertAlmostEqual(u_fem[1], u_correct[1])

    def test_3d(self):
        prm = c.Parameters(c.Constraint.FULL)
        mesh = df.UnitCubeMesh(5, 5, 5)

        u_bc = 42.0
        problem = c.MechanicsProblem(mesh, prm)
        bcs = []
        bcs.append(df.DirichletBC(problem.Vd.sub(0), 0, boundary.plane_at(0)))
        bcs.append(df.DirichletBC(problem.Vd.sub(0), u_bc, boundary.plane_at(1)))
        bcs.append(df.DirichletBC(problem.Vd.sub(1), 0, boundary.plane_at(0, "y")))
        bcs.append(df.DirichletBC(problem.Vd.sub(2), 0, boundary.plane_at(0, "z")))
        problem.set_bcs(bcs)

        u = problem.solve()

        xs = np.linspace(0, 1, 5)
        for x in xs:
            for y in xs:
                for z in xs:
                    u_fem = u((x, y, z))
                    # print(u_fem)
                    u_correct = (x * u_bc, -y * u_bc * prm.nu, -z * u_bc * prm.nu)
                    self.assertAlmostEqual(u_fem[0], u_correct[0])
                    self.assertAlmostEqual(u_fem[1], u_correct[1])
                    self.assertAlmostEqual(u_fem[2], u_correct[2])


if __name__ == "__main__":
    unittest.main()
