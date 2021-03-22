import constitutive as c
import unittest

class TestLinearElastic(unittest.TestCase):
    def test_1d(self):
        law = c.LinearElastic(2000, 0.2, c.Constraint.UNIAXIAL_STRAIN)
        stress, tangent = law.evaluate([42.])
        self.assertAlmostEqual(stress[0], 84000.)
        self.assertAlmostEqual(tangent[0,0], 2000.)

    def test_2d(self):
        law = c.LinearElastic(2000, 0.2, c.Constraint.PLANE_STRAIN)
        stress, tangent = law.evaluate([42., 0, 0])
        self.assertEqual(stress.shape[0], 3)
        self.assertEqual(tangent.shape, (3,3))

        self.assertEqual(c.q_dim(c.Constraint.PLANE_STRAIN), 3)
        self.assertEqual(c.g_dim(c.Constraint.PLANE_STRAIN), 2)


if __name__ == "__main__":
    unittest.main()

