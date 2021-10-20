import unittest
import numpy as np
import constitutive as c


class TestQValues(unittest.TestCase):
    def setUp(self):
        self.n = 3
        self.scalars = c.QValues(1)
        self.vectors = c.QValues(3)
        self.matrices = c.QValues(5,3)
        self.scalars.resize(self.n)
        self.vectors.resize(self.n)
        self.matrices.resize(self.n)

    def test_matrices_set_get(self):
        matrix = np.arange(15).reshape((5,3))
        for i in range(self.n):
            self.matrices.set(matrix, i)
            np.testing.assert_array_equal(self.matrices.get(i),matrix)

    def test_vectors_set_get(self):
        # in Eigen Vectors are \in \R^{n,1}
        vec = np.arange(3).reshape((3,1))
        for i in range(self.n):
            self.vectors.set(vec,i)
            np.testing.assert_array_equal(self.vectors.get(i),vec)

    def test_scalar_set_get(self):
        scalar = 42.
        for i in range(self.n):
            self.scalars.set(scalar,i)
            self.assertEqual(self.scalars.get_scalar(i), scalar)

    def test_internal_data(self):
        #Test if the internal data vector is in the correct layout
        matrix = np.arange(15).reshape((5,3))
        for i in range(self.n):
            self.matrices.set(matrix, i)

        np.testing.assert_array_equal(self.matrices.data,np.tile(matrix.flatten(),3))

if __name__ == "__main__":
    unittest.main()
