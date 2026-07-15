from __future__ import annotations

import dolfinx as df
import numpy as np
import pytest
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI

pytest.importorskip("dolfinx_external_operator")

from fenics_constitutive.external_operator import (
    ExternalOperatorProblem,
    IncrSmallStrainExternalOperator,
    grad_del_u_from_mandel_strain,
)
from fenics_constitutive.models import (
    LinearElasticityModel,
    MisesPlasticityLinearHardening3D,
    VonMises3D,
)
from fenics_constitutive.models.interfaces import StressStrainConstraint
from fenics_constitutive.models.utils import strain_from_grad_u
from fenics_constitutive.solver import IncrSmallStrainProblem

YOUNGS_MODULUS = 42.0
POISSONS_RATIO = 0.3

MISES_PARAM = {
    "p_ka": 175000.0,
    "p_mu": 80769.0,
    "p_y0": 1200.0,
    "p_y00": 2500.0,
    "p_w": 200.0,
}

MISES_PARAM_RUST = {
    "mu": np.array([MISES_PARAM["p_mu"]]),
    "kappa": np.array([MISES_PARAM["p_ka"]]),
    "y_0": np.array([MISES_PARAM["p_y0"]]),
    "h": np.array([MISES_PARAM["p_w"]]),
}


def _mises_law(model):
    try:
        return model(MISES_PARAM)
    except (KeyError, TypeError):
        return model(MISES_PARAM_RUST)


@pytest.mark.parametrize(
    "constraint",
    [
        StressStrainConstraint.UNIAXIAL_STRAIN,
        StressStrainConstraint.PLANE_STRAIN,
        StressStrainConstraint.FULL,
    ],
)
def test_grad_del_u_from_mandel_strain_roundtrip(constraint):
    rng = np.random.default_rng(42)
    n_points = 7
    grad_u = rng.random(n_points * constraint.geometric_dim**2)
    strain = strain_from_grad_u(grad_u, constraint)
    grad_u_sym = grad_del_u_from_mandel_strain(strain, constraint)
    np.testing.assert_allclose(
        strain_from_grad_u(grad_u_sym, constraint), strain, atol=1e-14
    )


def _uniaxial_stress_setup(mesh):
    """Unit cube under displacement-controlled uniaxial stress."""
    V = df.fem.functionspace(mesh, ("CG", 1, (3,)))
    u = df.fem.Function(V)

    tdim = mesh.topology.dim
    fdim = tdim - 1
    left_facets = df.mesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.isclose(x[0], 0.0)
    )
    right_facets = df.mesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.isclose(x[0], 1.0)
    )
    bottom_facets = df.mesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.isclose(x[1], 0.0)
    )
    side_facets = df.mesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.isclose(x[2], 0.0)
    )

    zero_scalar = df.fem.Constant(mesh, 0.0)
    scalar_x = df.fem.Constant(mesh, 0.0)
    bcs = [
        df.fem.dirichletbc(
            zero_scalar,
            df.fem.locate_dofs_topological(V.sub(0), fdim, left_facets),
            V.sub(0),
        ),
        df.fem.dirichletbc(
            scalar_x,
            df.fem.locate_dofs_topological(V.sub(0), fdim, right_facets),
            V.sub(0),
        ),
        df.fem.dirichletbc(
            zero_scalar,
            df.fem.locate_dofs_topological(V.sub(1), fdim, bottom_facets),
            V.sub(1),
        ),
        df.fem.dirichletbc(
            zero_scalar,
            df.fem.locate_dofs_topological(V.sub(2), fdim, side_facets),
            V.sub(2),
        ),
    ]
    return u, bcs, scalar_x


def test_elasticity_matches_incr_problem():
    """Linear elasticity: external-operator solve equals IncrSmallStrainProblem."""
    q_degree = 2

    def law():
        return LinearElasticityModel(
            parameters={"E": YOUNGS_MODULUS, "nu": POISSONS_RATIO},
            constraint=StressStrainConstraint.FULL,
        )

    mesh_ref = df.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    u_ref, bcs_ref, scalar_ref = _uniaxial_stress_setup(mesh_ref)
    problem_ref = IncrSmallStrainProblem(law(), u_ref, bcs_ref, q_degree=q_degree)
    solver_ref = NewtonSolver(MPI.COMM_WORLD, problem_ref)

    mesh_ext = df.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    u_ext, bcs_ext, scalar_ext = _uniaxial_stress_setup(mesh_ext)
    wrapper = IncrSmallStrainExternalOperator(law(), u_ext, q_degree=q_degree)
    problem_ext = ExternalOperatorProblem(wrapper, bcs_ext)
    solver_ext = NewtonSolver(MPI.COMM_WORLD, problem_ext)

    scalar_ref.value = scalar_ext.value = 0.01
    _, converged_ref = solver_ref.solve(u_ref)
    _, converged_ext = solver_ext.solve(u_ext)
    assert converged_ref and converged_ext
    problem_ref.update()
    problem_ext.update()

    np.testing.assert_allclose(u_ext.x.array, u_ref.x.array, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(
        wrapper.stress_0.x.array,
        problem_ref.stress_0.x.array,
        rtol=1e-8,
        atol=1e-10,
    )


@pytest.mark.parametrize("model", [VonMises3D, MisesPlasticityLinearHardening3D])
def test_mises_plasticity_matches_incr_problem(model):
    """Von Mises plasticity with history: load stepping into the plastic regime
    gives the same stress path and displacements as IncrSmallStrainProblem."""
    q_degree = 2
    n_steps = 20
    max_disp = 0.05

    mesh_ref = df.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    u_ref, bcs_ref, scalar_ref = _uniaxial_stress_setup(mesh_ref)
    problem_ref = IncrSmallStrainProblem(
        _mises_law(model), u_ref, bcs_ref, q_degree=q_degree
    )
    solver_ref = NewtonSolver(MPI.COMM_WORLD, problem_ref)

    mesh_ext = df.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    u_ext, bcs_ext, scalar_ext = _uniaxial_stress_setup(mesh_ext)
    wrapper = IncrSmallStrainExternalOperator(
        _mises_law(model), u_ext, q_degree=q_degree
    )
    problem_ext = ExternalOperatorProblem(wrapper, bcs_ext)
    solver_ext = NewtonSolver(MPI.COMM_WORLD, problem_ext)

    history = wrapper.history
    ref_history = problem_ref._history_0[0]
    assert history is not None
    assert ref_history is not None

    plastic = False
    for load in np.linspace(0, 1, num=n_steps + 1)[1:]:
        scalar_ref.value = scalar_ext.value = load * max_disp

        iter_ref, converged_ref = solver_ref.solve(u_ref)
        assert converged_ref
        problem_ref.update()

        iter_ext, converged_ext = solver_ext.solve(u_ext)
        assert converged_ext
        problem_ext.update()

        # A consistent tangent keeps Newton fast; exact counts can differ by
        # float noise at the convergence-tolerance boundary.
        assert iter_ext <= max(iter_ref, 5)

        np.testing.assert_allclose(
            u_ext.x.array, u_ref.x.array, rtol=1e-7, atol=1e-12
        )
        np.testing.assert_allclose(
            wrapper.stress_0.x.array,
            problem_ref.stress_0.x.array,
            rtol=1e-7,
            atol=1e-8,
        )
        for name, values in history.items():
            np.testing.assert_allclose(
                values,
                ref_history[name].x.array,
                rtol=1e-7,
                atol=1e-12,
            )
        if np.max(wrapper.stress_0.x.array) > MISES_PARAM["p_y0"]:
            plastic = True

    assert plastic, "test never entered the plastic regime"
