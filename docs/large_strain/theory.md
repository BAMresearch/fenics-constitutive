# Large-strain framework

This page describes the theory behind `LargeStrainProblem` and the Hencky models
(`HenckyElasticity3D`, `HenckyMises3D`, `HenckyDruckerPrager3D`,
`HenckyDruckerPragerHyperbolic3D`). The framework has two goals:

1. **No stress rates.** The stress is a function of the current deformation gradient and of
   history variables stored in the reference configuration. It is computed in closed form from a
   hyperelastic energy and is never obtained by integrating a stress rate over time. The model is
   therefore path-independent in the elastic range and thermodynamically consistent. The page
   [Stress rates and the Hencky model](stress_rates.ipynb) compares this with the hypoelastic
   approach of `CorotationalIncrSmallStrainProblem`.
2. **Reuse of the small-strain models.** For isotropic models, the return mapping in logarithmic
   strains has exactly the form of the small-strain return mapping. Every plasticity model that
   implements the Rust `Plasticity` trait is used unchanged for small and for large strains, and the
   same return-mapping solver is used in both cases.

The problem is formulated on the reference mesh, which is never moved. The unknown is the total
displacement $\mathbf u$.

## Kinematics

With the reference position $\mathbf X$ and the current position $\mathbf x = \mathbf X + \mathbf u$,

$$
\mathbf F = \mathbf I + \nabla\mathbf u, \qquad J = \det\mathbf F, \qquad \mathbf b = \mathbf F\mathbf F^T .
$$

The deformation gradient is split multiplicatively into an elastic and a plastic part,
$\mathbf F = \mathbf F^e\mathbf F^p$. The plastic part enters only through the inverse plastic right
Cauchy–Green tensor $\mathbf C_p^{-1} = (\mathbf F^{pT}\mathbf F^p)^{-1}$, which lives in the reference
configuration and is the main history variable. The elastic left Cauchy–Green tensor and the
logarithmic elastic strain are

$$
\mathbf b^e = \mathbf F^e\mathbf F^{eT} = \mathbf F\,\mathbf C_p^{-1}\mathbf F^T, \qquad
\boldsymbol\varepsilon_e = \tfrac12\ln\mathbf b^e .
$$

Under a superposed rigid rotation $\mathbf F\to\mathbf Q\mathbf F$, $\mathbf C_p^{-1}$ does not change and
$\mathbf b^e \to \mathbf Q\mathbf b^e\mathbf Q^T$, so all quantities derived below are objective without any
rotation of the history.

## Hencky elasticity

The free energy is quadratic in the logarithmic elastic strain,

$$
\psi(\boldsymbol\varepsilon_e) = \mu\,\lVert\mathrm{dev}\,\boldsymbol\varepsilon_e\rVert^2
  + \tfrac12\kappa\,(\mathrm{tr}\,\boldsymbol\varepsilon_e)^2,
$$

with the shear modulus $\mu$ and the bulk modulus $\kappa$. For an isotropic energy, the Kirchhoff
stress $\boldsymbol\tau = J\boldsymbol\sigma$ is

$$
\boldsymbol\tau = \frac{\partial\psi}{\partial\boldsymbol\varepsilon_e} = \mathbb C:\boldsymbol\varepsilon_e,
\qquad \mathbb C = 2\mu\,\mathbb P_{dev} + 3\kappa\,\mathbb P_{vol},
$$

which is coaxial with $\mathbf b^e$. $\mathbb C$ is the elastic tangent of the small-strain models. Without
plasticity ($\mathbf C_p^{-1} = \mathbf I$) this is $\boldsymbol\tau = \mathbb C:\ln\mathbf V$ with the left stretch
$\mathbf V = \mathbf b^{1/2}$. Since the stress follows from an energy, it returns to zero on every closed
deformation path and no work is done on closed cycles.

The Hencky model is equivalent to the hypoelastic equation
$\overset{\circ}{\boldsymbol\tau}{}^{\log} = \mathbb C:\mathbf d$ with the logarithmic stress rate (Xiao, Bruhns
and Meyers, 1997): the logarithmic rate is the only objective rate for which this equation is exactly
integrable, and its integral is $\boldsymbol\tau = \mathbb C:\ln\mathbf V$. The Hencky energy is not polyconvex,
which is irrelevant for the moderate elastic strains of metals, concrete or soils.

## Plasticity and the exponential map

The yield function $f$, the flow direction $\mathbf g$ and the hardening function $k$ are those of the
small-strain model, evaluated at the Kirchhoff stress (see [Yield in Cauchy stress](#yield-in-cauchy-stress)
for the alternative):

$$
f(\boldsymbol\tau,\alpha) \le 0, \qquad
-\tfrac12\,\mathcal L_v\mathbf b^e\,\mathbf b^{e\,-1} = \dot\lambda\,\mathbf g(\boldsymbol\tau,\alpha), \qquad
\dot\alpha = \dot\lambda\,k(\boldsymbol\tau,\alpha),
$$

where $\mathcal L_v\mathbf b^e = \mathbf F\,\dot{\mathbf C}_p^{-1}\mathbf F^T$ is the Lie derivative. Within a load step
from $t_n$ to $t_{n+1}$, the flow rule is integrated with the exponential map (Simo, 1992),

$$
\mathbf b^e_{n+1} = \exp\!\big(-2\Delta\lambda\,\mathbf g\big)\,\mathbf b^e_{tr}, \qquad
\mathbf b^e_{tr} = \mathbf F_{n+1}\,\mathbf C^{-1}_{p,n}\,\mathbf F^T_{n+1} .
$$

For isotropic models, $\mathbf g$ is coaxial with $\boldsymbol\tau$ and therefore with $\mathbf b^e_{tr}$, so the
matrix exponential and logarithm commute with the product. In logarithmic strains the update becomes
additive, and with $\boldsymbol\tau = \mathbb C:\boldsymbol\varepsilon_e$ the discrete equations are

$$
\boldsymbol\tau = \boldsymbol\tau_{tr} - \Delta\lambda\,\mathbb C:\mathbf g(\boldsymbol\tau,\alpha), \qquad
f(\boldsymbol\tau,\alpha) = 0, \qquad
\alpha = \alpha_n + \Delta\lambda\,k(\boldsymbol\tau,\alpha), \qquad
\boldsymbol\tau_{tr} = \mathbb C:\tfrac12\ln\mathbf b^e_{tr} .
$$

These are exactly the backward-Euler equations of the small-strain return mapping:

| | small strain | large strain (Hencky) |
|---|---|---|
| stress | Cauchy $\boldsymbol\sigma$ | Kirchhoff $\boldsymbol\tau$ |
| trial stress | $\boldsymbol\sigma_n + \mathbb C:\Delta\boldsymbol\varepsilon$ | $\mathbb C:\tfrac12\ln(\mathbf F\mathbf C_{p,n}^{-1}\mathbf F^T)$ |
| plastic strain increment | $\Delta\lambda\,\mathbf g$ | $\Delta\lambda\,\mathbf g$ (logarithmic) |
| history | $\alpha$, $\boldsymbol\varepsilon_p$ | $\alpha$, $\mathbf C_p^{-1}$ |

The same solver (`NewtonSolverStrain`) and the same `Plasticity` implementations are therefore used
for both. After convergence, the history is updated with

$$
\boldsymbol\varepsilon_e = \boldsymbol\varepsilon_{tr} - \Delta\lambda\,\mathbf g, \qquad
\mathbf C^{-1}_{p,n+1} = \mathbf F^{-1}_{n+1}\exp(2\boldsymbol\varepsilon_e)\,\mathbf F^{-T}_{n+1} .
$$

The initial history is $\mathbf C_p^{-1} = \mathbf I$, $\alpha = 0$, which is why large-strain models provide
`initial_history()` instead of starting from zeros. For coaxial deformation paths with fixed
principal axes, the large-strain model and the small-strain model driven with
$\Delta\boldsymbol\varepsilon = \ln(\boldsymbol\lambda_{n+1}/\boldsymbol\lambda_n)$ give the same stresses and hardening
variables to round-off.

### Yield in Cauchy stress

Optionally (`yield_stress="cauchy"`), the plasticity model is evaluated at the Cauchy stress
$\boldsymbol\sigma = \boldsymbol\tau/J$, while the elastic law stays $\boldsymbol\tau = \mathbb C:\boldsymbol\varepsilon_e$. Since
$J = \det\mathbf F_{n+1}$ is fixed during the return mapping, the model is wrapped (`CauchyScaled`) as

$$
\tilde f(\boldsymbol\tau,\alpha) = f(\boldsymbol\tau/J,\alpha), \qquad
\tilde{\mathbf g}(\boldsymbol\tau,\alpha) = \mathbf g(\boldsymbol\tau/J,\alpha), \qquad
\tilde k(\boldsymbol\tau,\alpha) = k(\boldsymbol\tau/J,\alpha),
$$

so that only the derivatives with respect to the stress change, by the factor $1/J$. Because the flow
direction is not scaled, $\Delta\lambda\,\mathbf g$ remains the logarithmic plastic strain increment and
$\alpha$ keeps its meaning, e.g. the equivalent plastic strain for von Mises. The two variants coincide
for isochoric deformations and differ when the volume changes, for example for pressure-dependent
models with dilatancy.

## Weak form

The balance of linear momentum in the current configuration $\Omega$ is
$\mathrm{div}_x\boldsymbol\sigma + \mathbf b = \mathbf 0$. Its weak form is pulled back to the reference
configuration $\Omega_0$ with $\nabla_x\mathbf v = \nabla\mathbf v\,\mathbf F^{-1}$ and $\mathrm dx = J\,\mathrm dX$:

$$
R(\mathbf u;\mathbf v) = \int_\Omega\boldsymbol\sigma:\nabla_x\mathbf v\,\mathrm dx - R_{ext}
 = \int_{\Omega_0}\boldsymbol\tau:\mathbf g_v\,\mathrm dX - R_{ext}, \qquad
\mathbf g_v = \nabla\mathbf v\,\mathbf F^{-1} .
$$

Since $\boldsymbol\tau:(\nabla\mathbf v\,\mathbf F^{-1}) = (\boldsymbol\tau\mathbf F^{-T}):\nabla\mathbf v$, this is identical to
$\int_{\Omega_0}\mathbf P:\nabla\mathbf v\,\mathrm dX$ with the first Piola–Kirchhoff stress $\mathbf P$. Only the symmetric
$\boldsymbol\tau$ (6 values in Mandel notation) and a $6\times6$ tangent are stored per quadrature point instead
of $\mathbf P$ (9 values) and $\partial\mathbf P/\partial\mathbf F$ (81 values).

### Linearization

Every variation of the deformation gradient can be written as $\delta\mathbf F = \mathbf l\,\mathbf F$ with

$$
\mathbf l = \nabla\delta\mathbf u\,\mathbf F^{-1} = \mathbf d + \mathbf w, \qquad
\mathbf d = \mathrm{sym}\,\mathbf l, \quad \mathbf w = \mathrm{skew}\,\mathbf l .
$$

The exact directional derivative of the residual is

$$
\delta R = \int_{\Omega_0}
\underbrace{(\mathbb a:\mathbf d):\mathbf g_v}_{\text{model}}
+ \underbrace{(\mathbf w\boldsymbol\tau - \boldsymbol\tau\mathbf w):\mathbf g_v}_{\text{spin}}
- \underbrace{\boldsymbol\tau:(\mathbf g_v\,\mathbf l)}_{\text{geometric}}\,\mathrm dX .
$$

The three terms have different origins:

- **Geometric term.** The spatial test gradient depends on $\mathbf u$ through $\mathbf F^{-1}$:
  $\delta(\mathbf F^{-1}) = -\mathbf F^{-1}\delta\mathbf F\,\mathbf F^{-1}$ gives $\delta\mathbf g_v = -\mathbf g_v\,\mathbf l$. This is pure
  kinematics.
- **Spin term.** $\delta\mathbf F = \mathbf w\mathbf F$ is an infinitesimal superposed rigid rotation. For every
  objective model, $\boldsymbol\tau(\mathbf Q\mathbf F) = \mathbf Q\,\boldsymbol\tau(\mathbf F)\,\mathbf Q^T$ with unchanged history, and
  differentiating along $\mathbf Q = \exp(\epsilon\mathbf w)$ gives $\delta\boldsymbol\tau = \mathbf w\boldsymbol\tau - \boldsymbol\tau\mathbf w$. The model
  does not need to provide this part.
- **Model term.** The constitutive model returns $\mathbb a = \partial\boldsymbol\tau/\partial\mathbf d$ as a $6\times6$ matrix in
  Mandel notation.

The split of $\delta\boldsymbol\tau$ into $\mathbb a:\mathbf d$ and a spin part resembles a Jaumann rate, but it is only a
reparametrization of the nine directions of $\delta\mathbf F$ for the Newton method. Nothing is integrated
over time, and an error in the tangent affects the convergence rate, not the solution.

### The tangent of the Hencky models

The model tangent is the product of a material and a kinematic part,

$$
\mathbb a = \mathbb D\,\mathbb E, \qquad
\mathbb D = \frac{\partial\boldsymbol\tau}{\partial\boldsymbol\varepsilon_{tr}}, \qquad
\mathbb E = \frac{\partial\boldsymbol\varepsilon_{tr}}{\partial\mathbf d} .
$$

$\mathbb D$ is the tangent of the small-strain model: $\mathbb D = \mathbb C$ in elastic steps, and the algorithmic
tangent $\mathbb D = [\mathbf J_N^{-1}]_{\tau\tau}\,\mathbb C$ in plastic steps, where $\mathbf J_N$ is the Jacobian of the
return-mapping equations with respect to $(\boldsymbol\tau, \Delta\lambda, \alpha)$ at the converged state.

$\mathbb E$ follows from the trial strain. With $\mathbf C^{-1}_{p,n}$ fixed,
$\delta\mathbf b^e_{tr} = \mathbf l\,\mathbf b^e_{tr} + \mathbf b^e_{tr}\mathbf l^T$; its symmetric part
$\mathbf d\,\mathbf b^e_{tr} + \mathbf b^e_{tr}\mathbf d$ is mapped through the derivative of $\tfrac12\ln$ (Daleckii–Krein
formula), while the skew part only rotates $\boldsymbol\varepsilon_{tr}$. With
$\mathbf b^e_{tr} = \mathbf Q\,\mathrm{diag}(\lambda_i)\,\mathbf Q^T$,

$$
\delta\boldsymbol\varepsilon_{tr} = \mathbf Q\big(\Gamma\circ\mathbf Q^T(\mathbf d\,\mathbf b^e_{tr} + \mathbf b^e_{tr}\mathbf d)\mathbf Q\big)\mathbf Q^T,
\qquad
\Gamma_{ij} = \frac{\ln\lambda_i - \ln\lambda_j}{2(\lambda_i - \lambda_j)}, \quad \Gamma_{ii} = \frac{1}{2\lambda_i} .
$$

In the eigenbasis of $\mathbf b^e_{tr}$, $\mathbb E$ is diagonal: it is the identity on the normal components and
scales the shear components by $(\lambda_i + \lambda_j)(\ln\lambda_i - \ln\lambda_j)/(2(\lambda_i - \lambda_j)) \ge 1$. For
$\mathbf F\to\mathbf I$, $\mathbb E\to\mathbb I$, the spin and geometric terms vanish, and the small-strain tangent is recovered.

With yield in Cauchy stress, $\boldsymbol\tau$ also depends on $J$, and $\delta J = J\,\mathrm{tr}\,\mathbf d$ adds the rank-one
term $\big(J\,\partial\boldsymbol\tau/\partial J\big)\otimes\mathbf I$ to $\mathbb a$. $J\,\partial\boldsymbol\tau/\partial J$ follows from implicit
differentiation of the converged return mapping with the same matrix $\mathbf J_N^{-1}$.

The tangent is not symmetric in general. $\mathbb D$ is nonsymmetric for non-associated flow, and so is the
whole Jacobian then; for Hencky elasticity and associated flow it is symmetric.

## Boundary conditions and loads

- Dirichlet conditions prescribe the total displacement $\mathbf u = \mathbf x - \mathbf X$ at reference positions,
  e.g. $\mathbf u = (\mathbf R - \mathbf I)\mathbf X$ for a rigid rotation $\mathbf R$.
- `external_forces` are integrated over the reference mesh: `ufl.dx` and `ufl.ds` act on $\Omega_0$ and
  $\partial\Omega_0$, and `ufl.FacetNormal` is the reference normal $\mathbf N$. Loads must therefore be nominal
  quantities. A body force $\mathbf b$ per current volume becomes $\int_{\Omega_0}J\,\mathbf b\cdot\mathbf v\,\mathrm dX$, and a
  traction $\mathbf t_0$ per reference area enters as $\int_{\partial\Omega_0}\mathbf t_0\cdot\mathbf v\,\mathrm dA$.
- Deformation-dependent loads such as a pressure $p$ on the current surface are pulled back with Nanson's
  formula $\mathbf n\,\mathrm da = J\mathbf F^{-T}\mathbf N\,\mathrm dA$:
  $-\int_{\partial\Omega_0}p\,J\mathbf F^{-T}\mathbf N\cdot\mathbf v\,\mathrm dA$ with $\mathbf F = \mathbf I + \nabla\mathbf u$ written in terms of
  `u`. Their contribution to the Jacobian is currently not included, which slows down the convergence
  but does not change the solution.

## Implementation

### Model interface

A large-strain model implements `LargeStrainModel` in Python, or `FiniteStrainConstitutiveModelFn` in
Rust. All arrays are flat and hold all quadrature points:

| argument | content per quadrature point |
|---|---|
| `F` | $\mathbf F_{n+1}$, 9 values, row-major: `F[3*i + J]` $= F_{iJ}$ |
| `tau` | output, $\boldsymbol\tau_{n+1}$ in Mandel notation $[\tau_{11},\tau_{22},\tau_{33},\sqrt2\tau_{12},\sqrt2\tau_{13},\sqrt2\tau_{23}]$ |
| `tangent` | output, $\mathbb a$ as a row-major $6\times6$ Mandel matrix, entry $(i,j) = \partial\tau_i/\partial d_j$; `None` to skip |
| `history` | dict of arrays, state at $t_n$ on input and at $t_{n+1}$ on output |

`history_dim` declares the history fields, as for small-strain models, and `initial_history()`
returns the history of one quadrature point in the undeformed state. Models raise `ValueError` if a
point cannot be evaluated, e.g. for $\det\mathbf F\le0$ or if the return mapping fails. The model has to be
objective: history variables must be stored in a configuration that superposed rigid rotations do
not affect.

The Hencky plasticity models store one history field `"history"` with 7 values per point: $\mathbf C_p^{-1}$ in
Mandel notation followed by $\alpha$.

### Rust side

- `comfe-rs/src/finite_strain/tensor.rs`: spectral functions of symmetric $3\times3$ matrices. The
  eigen decomposition uses the cyclic Jacobi method, which reconstructs nearly degenerate matrices to
  machine precision. $\tfrac12\ln\mathbf b$ is computed from the eigenvalues $x_i$ of $\mathbf b - \mathbf I$ with
  `ln_1p`, which is accurate close to the undeformed state, and the divided differences $\Gamma$ are
  evaluated stably for (nearly) equal eigenvalues.
- `comfe-rs/src/finite_strain/hencky.rs`: `HenckyKinematics` (trial strain and $\mathbb E$),
  `HenckyElasticity3D` and `HenckyPlasticity3D<N_PARAMETERS, PARAMETERS, MODEL, CAUCHY_YIELD>`.
- `comfe-rs/src/finite_strain/cauchy.rs`: `CauchyScaled`, the wrapper for yield in Cauchy stress.
- `comfe-rs/src/plasticity/`: the unchanged `Plasticity` models and the return-mapping solver
  `NewtonSolverStrain`, shared with the small-strain models.

A new isotropic plasticity model needs only an implementation of `Plasticity`; one line in
`bindings/src/lib.rs` (`implement_python_finite_strain_model!`) and a wrapper class in
`models/rust_models.py` make it available as a large-strain model.

### Problem class

`LargeStrainProblem` stores $\mathbf F$, $\boldsymbol\tau$ and $\mathbb a$ as quadrature functions
(`deformation_gradient`, `tau`, `tangent`) and the history of every law as arrays per quadrature
point (`history(i)`). Before each assembly, `form` interpolates $\mathbf F$, copies the committed history to
the trial history and evaluates all laws. `update()` commits the converged step, `reset()` restores the
last committed displacement. `cauchy_stress()` and `piola_stress()` return the other stress measures as
ufl expressions.

If a law raises `ValueError` on any MPI rank, all ranks raise `ConstitutiveError` together, so a failed
step does not deadlock a parallel run. `solve_incrementally` uses this to halve the load step until the
step converges.

### Example

```python
import dolfinx as df
import numpy as np
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI

from fenics_constitutive.models import HenckyMises3D
from fenics_constitutive.solver import LargeStrainProblem, solve_incrementally

mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)
V = df.fem.functionspace(mesh, ("Lagrange", 1, (3,)))
u = df.fem.Function(V)

# rollers on x=0, y=0, z=0 and a prescribed displacement on x=1
fdim = mesh.topology.dim - 1
zero = df.fem.Constant(mesh, 0.0)
stretch = df.fem.Constant(mesh, 0.0)
bcs = []
for axis in range(3):
    facets = df.mesh.locate_entities_boundary(mesh, fdim, lambda x, a=axis: np.isclose(x[a], 0.0))
    dofs = df.fem.locate_dofs_topological(V.sub(axis), fdim, facets)
    bcs.append(df.fem.dirichletbc(zero, dofs, V.sub(axis)))
facets = df.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], 1.0))
dofs = df.fem.locate_dofs_topological(V.sub(0), fdim, facets)
bcs.append(df.fem.dirichletbc(stretch, dofs, V.sub(0)))

parameters = {
    "mu": np.array([80.77]),
    "kappa": np.array([166.67]),
    "y_0": np.array([0.2]),
    "h": np.array([10.0]),
}
law = HenckyMises3D(parameters, yield_stress="kirchhoff")
problem = LargeStrainProblem(law, u, bcs, q_degree=1)
solver = NewtonSolver(MPI.COMM_WORLD, problem)

def apply_load(factor: float) -> None:
    stretch.value = 0.5 * factor  # 50 % elongation

iterations = solve_incrementally(problem, solver, apply_load, n_steps=20)
alpha = problem.history()["history"][:, 6]  # equivalent plastic strain per quadrature point
```

## Limitations

- Only 3D problems are supported. Plane strain is possible by padding $\mathbf F$ with $F_{33} = 1$ and is not
  implemented yet.
- Only isotropic plasticity models with a scalar hardening variable are supported. Kinematic hardening
  needs a back stress that is not coaxial with $\mathbf b^e$; then the exponential map no longer reduces to the
  small-strain return mapping, and a different formulation is required.
- Gradient-enhanced (nonlocal) models are not available for large strains yet.
- Deformation-dependent external loads are not linearized.

## Verification

The implementation is covered by the following tests:

- Rust (`comfe-rs/src/finite_strain/`): tensor functions against finite differences, including
  repeated eigenvalues and states close to the identity; the analytic Hencky stress; the tangent against
  finite differences along all nine directions $\delta\mathbf F = \mathbf l\mathbf F$ for elastic and plastic points of all
  models and both yield variants; coaxial equivalence with the small-strain models up to stretches of 2
  and 0.5; objectivity; the small-strain limit; the yield condition in Cauchy stress.
- Python (`tests/solver/test_large_strain.py`, also with two MPI ranks): uniaxial tension and
  compression to 50 % against analytic solutions for both yield variants with at most six Newton
  iterations per step; rotation of the Cauchy stress under a superposed rigid rotation; agreement with
  `IncrSmallStrainProblem` at small strains; equality of the residual with
  $\int_{\Omega_0}\mathbf P:\nabla\mathbf v\,\mathrm dX$; the assembled Jacobian against finite differences of the
  residual; several laws on cell subsets; step cutting and collective error handling.

## References

- A. L. Eterovic, K.-J. Bathe. A hyperelastic-based large strain elasto-plastic constitutive formulation
  with combined isotropic-kinematic hardening using the logarithmic stress and strain measures.
  *International Journal for Numerical Methods in Engineering* 30 (1990) 1099–1114.
- J. C. Simo. Algorithms for static and dynamic multiplicative plasticity that preserve the classical
  return mapping schemes of the infinitesimal theory. *Computer Methods in Applied Mechanics and
  Engineering* 99 (1992) 61–112.
- J. C. Simo, T. J. R. Hughes. *Computational Inelasticity*. Springer, 1998.
- E. A. de Souza Neto, D. Perić, D. R. J. Owen. *Computational Methods for Plasticity*. Wiley, 2008.
- H. Xiao, O. T. Bruhns, A. Meyers. Hypo-elasticity model based upon the logarithmic stress rate.
  *Journal of Elasticity* 47 (1997) 51–68.
