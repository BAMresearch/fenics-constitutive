from fenics_constitutive import Constraint, strain_from_grad_u
import numpy as np


def test_strain_from_grad_u():
    grad_u = np.array([[1.0]])
    constraint = Constraint.UNIAXIAL_STRAIN
    strain = strain_from_grad_u(grad_u, constraint)
    assert np.allclose(strain, np.array([1.0]))
    constraint = Constraint.UNIAXIAL_STRESS
    strain = strain_from_grad_u(grad_u, constraint)
    assert np.allclose(strain, np.array([1.0]))
    grad_u = np.array([[1.0, 2.0], [3.0, 4.0]])
    constraint = Constraint.PLANE_STRAIN
    strain = strain_from_grad_u(grad_u, constraint)
    assert np.allclose(strain, np.array([1.0, 4.0, 0.0, 0.5 * (4.0 + 1.0) * 2**0.5]))
    constraint = Constraint.PLANE_STRESS
    strain = strain_from_grad_u(grad_u, constraint)
    assert np.allclose(strain, np.array([1.0, 4.0, 0.0, 0.5 * (4.0 + 1.0) * 2**0.5]))
    grad_u = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    constraint = Constraint.FULL
    strain = strain_from_grad_u(grad_u, constraint)
    assert np.allclose(
        strain,
        np.array(
            [
                1.0,
                5.0,
                9.0,
                0.5 * (2.0 + 4.0) * 2**0.5,
                0.5 * (6.0 + 8.0) * 2**0.5,
                0.5 * (3.0 + 7.0) * 2**0.5,
            ]
        ),
    )
