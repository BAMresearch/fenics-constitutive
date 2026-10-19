# Plan: Multiplicative finite-strain plasticity ($\boldsymbol F = \boldsymbol F^e \boldsymbol F^p$) in fenics-constitutive

**Goal:** finite-strain elastoplasticity based on the multiplicative split $\boldsymbol F = \boldsymbol F^e \boldsymbol F^p$, reusing the
existing Rust plasticity code with as few changes as possible:

- the `Plasticity` trait implementations,
- the `IsotropicPlasticityModel3D` return mapping,
- the gradient-enhanced `IsotropicGradientPlasticityModel3D`.

Interfaces may be adapted.

**Companion document:** [FINITE_STRAIN_PLAN.md](FINITE_STRAIN_PLAN.md), the additive log-strain-space variant. Section 8 compares the two.

**Shared prerequisites** (listed in the companion document):
- the Mandel-ordering bug in the corotational code,
- the `nabla_grad` vs `grad` convention,
- the tangent transpose convention.

**Recommendation in one sentence:** implement the Simo (1992) / Eterovic–Bathe exponential-map
algorithm. For isotropic elasticity and isotropic yield, its return mapping in the elastic
log-strain $\boldsymbol\varepsilon^e = \tfrac12 \ln \boldsymbol b^e$ is *algebraically identical* to the small-strain return mapping,
so the Rust models only need to be called with a different trial state.

---

## 1. Theory

### 1.1 Kinematics and constitutive assumptions

- **Multiplicative split:** $\boldsymbol F = \boldsymbol F^e \boldsymbol F^p$, with the elastic left Cauchy–Green tensor $\boldsymbol b^e = \boldsymbol F^e \boldsymbol F^{e\,T}$ (spatial, symmetric).
- **Elastic log strain:** $\boldsymbol\varepsilon^e = \tfrac12 \ln \boldsymbol b^e$.
- **Hencky hyperelasticity:**

$$
  \psi(\boldsymbol\varepsilon^e) = \mu \,\lVert \operatorname{dev}\boldsymbol\varepsilon^e \rVert^2 + \tfrac12 \kappa \,(\operatorname{tr}\boldsymbol\varepsilon^e)^2,
  \qquad
  \boldsymbol\tau = \frac{\partial\psi}{\partial\boldsymbol\varepsilon^e} = \mathbb C : \boldsymbol\varepsilon^e .
$$

  This is exactly the linear isotropic elastic law already in the Rust models, with $\boldsymbol\sigma$ reinterpreted as the Kirchhoff stress $\boldsymbol\tau$.
- **Yield and flow:** the yield function $f(\boldsymbol\tau, \alpha)$ and flow potential $g(\boldsymbol\tau, \alpha)$ are the existing small-strain functions, evaluated with Kirchhoff stress.
- **Isotropy is required** for elasticity and yield, and plastic spin is assumed zero. All current models (Mises, Drucker–Prager classic/hyperbolic) satisfy this: they are functions of $I_1$ and $J_2$ with isotropic hardening.
- **Flow rule** (Lie derivative):

$$
  -\tfrac12\, \mathcal L_v \boldsymbol b^e \, (\boldsymbol b^e)^{-1} = \dot\lambda \,\frac{\partial g}{\partial \boldsymbol\tau}.
$$

### 1.2 Time discretization (exponential map), per quadrature point

Given $\boldsymbol F_n$, $\boldsymbol F_{n+1}$, the stored $\boldsymbol\varepsilon^e_n$, and the internal variables $\alpha_n$:

1. **Relative deformation gradient:** $\boldsymbol f = \boldsymbol F_{n+1} \boldsymbol F_n^{-1}$.
2. **Elastic trial predictor:**

$$
   \boldsymbol b^e_\text{tr} = \boldsymbol f \, \exp(2\boldsymbol\varepsilon^e_n) \, \boldsymbol f^T,
   \qquad
   \boldsymbol\varepsilon^e_\text{tr} = \tfrac12 \ln \boldsymbol b^e_\text{tr}.
$$

3. **Return mapping in log-strain space:**

$$
   \boldsymbol\varepsilon^e_{n+1} = \boldsymbol\varepsilon^e_\text{tr} - \Delta\lambda \,\frac{\partial g}{\partial\boldsymbol\tau},
   \qquad
   \boldsymbol\tau = \mathbb C : \boldsymbol\varepsilon^e_{n+1},
   \qquad
   f(\boldsymbol\tau, \alpha) = 0 .
$$

   Equivalently, $\boldsymbol\tau = \boldsymbol\tau_\text{tr} - \Delta\lambda\, \mathbb C : \partial g/\partial\boldsymbol\tau$ with $\boldsymbol\tau_\text{tr} = \mathbb C : \boldsymbol\varepsilon^e_\text{tr}$.

   **This is identical to `IsotropicPlasticityModel3D::evaluate` with `sigma_0 = 0` and `del_strain` $= \boldsymbol\varepsilon^e_\text{tr}$.**
   - The code computes `sigma_tr = C·del_eps + sigma_0`.
   - The residual is `res_sigma` $= \boldsymbol\sigma - \boldsymbol\sigma_\text{tr} + \Delta\lambda\, \mathbb C\, \boldsymbol g$.
   - The hardening update is unchanged.
   - `sigma_0` is passed to `set_model_state`, but no implementation uses it (checked: `isotropic_mises_plasticity.rs`, `drucker_prager_classic.rs`, `drucker_prager_hyperbolic.rs`).
   - The same holds for `MisesPlasticity3D` (radial return): with zero input stress, $p_1 = \kappa \operatorname{tr}\boldsymbol\varepsilon^e_\text{tr}$ and $\boldsymbol s_\text{tr} = 2\mu \operatorname{dev}\boldsymbol\varepsilon^e_\text{tr}$.
4. **Elastic strain update:** $\boldsymbol\varepsilon^e_{n+1} = \mathbb C^{-1} : \boldsymbol\tau_{n+1}$.
   - The Hencky law is linear and invertible, so no plastic-strain bookkeeping is needed.
   - The stored state is $\boldsymbol\varepsilon^e$, which starts at **zero**, so the existing zero-initialized `History` works.
   - For models that return a damaged stress, see §5.2.

The returned algorithmic tangent $\mathbb D = \partial\boldsymbol\tau_{n+1}/\partial\boldsymbol\varepsilon^e_\text{tr}$ is exactly what the Rust models already compute: `inverse[0:6,0:6]·C` in `general.rs`, and the closed form in `mises_plasticity.rs`.

### 1.3 Consistent spatial tangent

Let $\boldsymbol l = \operatorname{Grad}(\Delta\boldsymbol u)\, \boldsymbol F_{n+1}^{-1}$ be the spatial velocity gradient of a perturbation. Then

$$
\mathrm d\boldsymbol b^e_\text{tr} = \boldsymbol l\, \boldsymbol b^e_\text{tr} + \boldsymbol b^e_\text{tr}\, \boldsymbol l^T =: \mathbb B : \boldsymbol l,
\qquad
\mathbb B_{ijkl} = \delta_{ik}\, (b^e_\text{tr})_{jl} + \delta_{jk}\, (b^e_\text{tr})_{il},
$$

$$
\mathrm d\boldsymbol\varepsilon^e_\text{tr} = \tfrac12\, \mathbb L : \mathrm d\boldsymbol b^e_\text{tr},
\qquad
\mathbb L = \left.\frac{\partial \ln \boldsymbol b}{\partial \boldsymbol b}\right|_{\boldsymbol b^e_\text{tr}},
$$

$$
\boxed{\;\mathrm d\boldsymbol\tau = \mathbb A : \boldsymbol l, \qquad \mathbb A = \tfrac12\, \mathbb D : \mathbb L : \mathbb B\;}
$$

**Storage shape.** The rows of $\mathbb A$ are symmetric (6 Mandel components). The columns are the full, non-symmetric $\boldsymbol l$ (9 components). So $\mathbb A$ is stored as a **$6\times 9$** matrix per quadrature point. It depends on the spin part of $\boldsymbol l$, so it does not have minor symmetry in $kl$.

**Derivative of the logarithm.** Only the **first** derivative is needed; the additive log-space plan needs the second one, so this is a real simplification. Use the Daleckii–Krein formula with $\boldsymbol b^e_\text{tr} = \boldsymbol Q \boldsymbol\Lambda \boldsymbol Q^T$:

$$
\mathbb L[\boldsymbol H] = \boldsymbol Q \left( \boldsymbol\Gamma \circ (\boldsymbol Q^T \boldsymbol H \boldsymbol Q) \right) \boldsymbol Q^T,
\qquad
\Gamma_{ab} =
\begin{cases}
\dfrac{\ln\lambda_a - \ln\lambda_b}{\lambda_a - \lambda_b}, & \lambda_a \neq \lambda_b,\\[2ex]
\dfrac{1}{\lambda_a}, & \lambda_a = \lambda_b .
\end{cases}
$$

Repeated eigenvalues are always present at $\boldsymbol F = \boldsymbol I$. They need a tolerance-based switch to the limit value.

**Reference:** de Souza Neto, Perić & Owen, *Computational Methods for Plasticity* (2008), Ch. 14. There the spatial modulus is $\mathsf a = \frac{1}{2J}\, \mathbb D : \mathbb L : \mathbb B - \sigma_{il}\delta_{jk}$. Our $\mathbb A$ is the material part times $J$; the geometric part $-\sigma_{il}\delta_{jk}$ is handled in UFL (§1.4).

### 1.4 Weak form (reference configuration, mesh never moves)

With $\boldsymbol l(\boldsymbol w) := \operatorname{Grad}\boldsymbol w\; \boldsymbol F^{-1}$:

$$
R(\boldsymbol u; \boldsymbol v) = \int_{\Omega_0} \boldsymbol\tau : \boldsymbol l(\boldsymbol v) \,\mathrm dX - f_\text{ext}(\boldsymbol v),
$$

$$
\mathrm dR(\boldsymbol u; \boldsymbol v)[\Delta\boldsymbol u]
= \int_{\Omega_0} \big(\mathbb A : \boldsymbol l(\Delta\boldsymbol u)\big) : \boldsymbol l(\boldsymbol v) \,\mathrm dX
- \int_{\Omega_0} \boldsymbol\tau : \big(\boldsymbol l(\boldsymbol v)\, \boldsymbol l(\Delta\boldsymbol u)\big) \,\mathrm dX .
$$

- The second term comes from $\mathrm d(\boldsymbol F^{-1}) = -\boldsymbol F^{-1} \operatorname{Grad}\Delta\boldsymbol u\, \boldsymbol F^{-1}$; UFL provides it exactly.
- Only $\boldsymbol\tau$ (Mandel, 6) and $\mathbb A$ ($6\times 9$) come from the constitutive model.
- Because everything is in the reference configuration, there is no `MeshUpdater` and no isoparametric restriction.

---

## 2. What is reused, what changes (Rust)

| Code | Status |
|---|---|
| `plasticity/drucker_prager_classic.rs`, `drucker_prager_hyperbolic.rs`, `isotropic_mises_plasticity.rs` (the `Plasticity` impls) | **unchanged** |
| `plasticity/general.rs`: `IsotropicPlasticityModel3D` return mapping and algorithmic tangent | **unchanged**: called with `stress = 0`, `del_strain` $=\boldsymbol\varepsilon^e_\text{tr}$. Optional cleanup: extract the Newton loop into a `return_map(trial, alpha_n) -> (tau, del_lambda, alpha, D)` function, so the "stress = 0" trick isn't needed and the unused `plastic_strain` update can be skipped. |
| `plasticity/general_gradient_enhanced.rs` (branch `87-half-gradient-enahnced-plasticity`) | **unchanged**: same trick, see §5 |
| `mises_plasticity.rs`, `linear_elasticity.rs` | **unchanged** |
| `mandel.rs`, `consts.rs` | reused, plus a few new helpers |
| `interfaces.rs` | **extended**: a new trait `HenckyElastic` (§2.1), and new traits `FiniteStrainConstitutiveModelFn` / `FiniteStrainGradientConstitutiveModelFn` with their evaluation loops (§2.2, §5.5) |
| new `comfe-rs/src/finite_strain/` | `spectral.rs` (symmetric 3×3 eigendecomposition, $\ln$, $\exp$, Daleckii–Krein $\mathbb L$), `multiplicative.rs` (generic incremental wrapper), `total.rs` (generic total-strain wrapper, §5.3), `tangent.rs` ($\mathbb B$ and the $6\times9$ assembly) |

### 2.1 Interface addition: recovering the elastic strain

Step 4 of §1.2 needs $\boldsymbol\varepsilon^e_{n+1}$ from $\boldsymbol\tau_{n+1}$. The generic `ConstitutiveModelFn` does not expose the compliance, so add a small trait:

```rust
/// Models whose elastic part is linear and isotropic, i.e. Hencky-hyperelastic when σ is read as τ.
pub trait HenckyElastic<const HISTORY: usize, const PARAMETERS: usize> {
    /// Elastic log strain from the (returned) Kirchhoff stress. Default case: ℂ⁻¹ τ.
    /// Damage models return ℂ⁻¹ τ / (1 − ω), using ω from the history.
    fn elastic_strain(tau: &SVector<f64, 6>, history: &[f64; HISTORY], parameters: &[f64; PARAMETERS]) -> SVector<f64, 6>;
}
```

Implementations, each a few lines:

- `IsotropicPlasticityModel3D<…, M>`: `M::new(p).elastic_tangent_inv() * tau`. `elastic_tangent_inv` already exists in the `Plasticity` trait.
- `MisesPlasticity3D` and `LinearElasticity3D`: `isotropic_elastic_tangent_inv(mu, kappa) * tau`.
- Models that return effective stresses but carry `omega` in their history (e.g. `IsotropicGradientPlasticityModel3D` returns `result.sigma` and only stores `omega`): the same compliance, but check each model.

Implementing the trait doubles as an **explicit opt-in**. Only models known to be isotropic and Hencky-compatible can be wrapped, which protects against wrapping models where this kinematics is invalid:
- anisotropy,
- kinematic hardening with a back-stress,
- viscoelastic stress history.

### 2.2 Generic incremental wrapper

```rust
pub trait FiniteStrainConstitutiveModelFn<const N_HISTORY, const HISTORY, const N_PARAMETERS, const PARAMETERS> {
    type History; type Parameters;
    fn evaluate(
        time: f64, del_time: f64,
        f_0: &[[f64; 3]; 3], f_1: &[[f64; 3]; 3],          // F_n, F_{n+1}
        tau: &mut [f64; 6],                                // out: Kirchhoff stress, Mandel
        tangent: Option<&mut [[f64; 9]; 6]>,               // out: 𝔸 = dτ/dl
        elastic_log_strain: &mut [f64; 6],                 // in/out: εᵉ (Mandel)
        history: &mut [f64; HISTORY],                      // inner model history
        parameters: &[f64; PARAMETERS],
    );
}

pub struct Multiplicative<M>(PhantomData<M>);
impl<M> FiniteStrainConstitutiveModelFn<NH, H, NP, P> for Multiplicative<M>
where M: ConstitutiveModelFn<6, NH, H, NP, P> + HenckyElastic<H, P> { … }
```

**Where $\boldsymbol\varepsilon^e$ is stored.** It is a **separate argument and history entry** (`"elastic_log_strain"`). Concatenating it into the inner history would need `H + 6` in a const generic, which requires the unstable `generic_const_exprs`.

**Algorithm per quadrature point:**
1. $\boldsymbol f = \boldsymbol F_1 \boldsymbol F_0^{-1}$
2. $\boldsymbol b^e_\text{tr} = \boldsymbol f \exp(2\boldsymbol\varepsilon^e)\boldsymbol f^T$
3. Eigendecomposition of $\boldsymbol b^e_\text{tr}$ $\to$ $\boldsymbol\varepsilon^e_\text{tr}$ and $\mathbb L$
4. `M::evaluate(t, dt, εᵉ_tr, stress = 0, Some(D), history, p)`
5. $\boldsymbol\varepsilon^e \leftarrow$ `M::elastic_strain(τ, history, p)`
6. If a tangent is requested: $\mathbb A = \tfrac12 \mathbb D \mathbb L \mathbb B$

The $\exp(2\boldsymbol\varepsilon^e_n)$ in step 2 reuses the eigenvectors of $\boldsymbol\varepsilon^e_n$, so it costs one extra eigendecomposition. Storing $\boldsymbol b^e_n$ instead of $\boldsymbol\varepsilon^e_n$ would avoid it, but `History` would then need identity initialization.

**Tangent layout.** `general.rs` transposes the tangent before writing it out (`// TODO: move the transpose to the python bindings`), and nalgebra's `.data.0` is column-major. Read the inner $\mathbb D$ back with the matching layout. Cover this with a test using non-associated Drucker–Prager, whose $\mathbb D$ is non-symmetric.

### 2.3 Bindings

In `bindings/src/lib.rs`, add a macro along the lines of
`implement_multiplicative_python_model!(m, PyMultiplicativeIsotropicMises3D, IsotropicPlasticityModel3D<4,4,IsotropicMises3D>, StressStrainConstraint::FULL)`.

- It generates `evaluate(t, dt, grad_u, grad_del_u, tau, tangent, history)`.
- It builds $\boldsymbol F_1 = \boldsymbol I + (\nabla\boldsymbol u)^T$ and $\boldsymbol F_0 = \boldsymbol I + (\nabla\boldsymbol u - \nabla\Delta\boldsymbol u)^T$. Mind the `nabla_grad` transpose.
- `history_dim` returns `{"history": H, "elastic_log_strain": 6}`.

**One line per model** then exposes the finite-strain version.

---

## 3. Python side

### 3.1 Interface (`models/interfaces.py`)

```python
class IncrFiniteStrainModel(ABC):
    @abstractmethod
    def evaluate(
        self,
        t: float,
        del_t: float,
        grad_u: np.ndarray,          # total displacement gradient at t_{n+1}, raw gdim×gdim per QP
        grad_del_u: np.ndarray,      # increment (F_n is reconstructed from both)
        stress: np.ndarray,          # out: Kirchhoff stress τ (Mandel)
        tangent: np.ndarray | None,  # out: 𝔸 = dτ/dl, shape (n_qp, 6, 9); None => skip
        history: dict[str, np.ndarray] | None,
    ) -> None: ...
```

This $(\boldsymbol\tau, \partial\boldsymbol\tau/\partial\boldsymbol l)$ interface is **general**, not specific to plasticity:
- Any finite-strain model can implement it, such as hyperelastic Neo-Hooke in Python.
- The additive log-space wrapper from the other plan could also be written against it.
- One solver can therefore serve both approaches.

### 3.2 Solver (`solver/_finite_strain_solver.py`)

`IncrFiniteStrainProblem(NonlinearProblem)`, parallel to `IncrSmallStrainProblem`:

```python
F    = ufl.Identity(d) + ufl.grad(u)
Finv = ufl.inv(F)
def lx(w):  return ufl.grad(w) * Finv                   # spatial gradient, 3×3
tau_t = mandel_to_tensor(tau)                           # new UFL helper in solver/utils.py

R = ufl.inner(tau_t, lx(v)) * dxm - sum(external_forces)
J = ufl.inner(mandel_to_tensor(ufl.dot(A, flatten(lx(du)))), lx(v)) * dxm \
  - ufl.inner(tau_t, lx(v) * lx(du)) * dxm
```

- **`flatten` ordering:** `flatten(·)` must use the same 9-component ordering as the Rust $\mathbb A$ columns. Define it once in `solver/utils.py`.
- **Spaces:** the stress space (6) is reused from `ElementSpaces`. Add `ElementSpaces.finite_strain_tangent_space(mesh)` with value shape `(6, 9)`.
- **Kinematics:**
  - Extend `IncrementalDisplacement` with a total-gradient `Expression`.
  - Add `FiniteStrainLawOnSubMesh(LawOnSubMesh)` with one extra quadrature function for `grad_u`.
  - `History`, `build_subspace_map` and multi-material submeshes are reused unchanged.
- **Postprocessing:**
  - Cauchy stress is $\boldsymbol\sigma = \boldsymbol\tau / J$; add a helper for sensors.
  - The equivalent plastic strain `alpha`, if present in the inner history, keeps its meaning.

### 3.3 Python wrapper (`models/rust_models.py`)

Add a `fenics_constitutive_finite_strain_wrapper` decorator, analogous to the existing one but checking against `IncrFiniteStrainModel`. Then add classes such as `MultiplicativeIsotropicMises3D` and `MultiplicativeDruckerPragerHyperbolic3D`.

### 3.4 Optional: numpy prototype first

Before writing any Rust, a numpy wrapper `MultiplicativeAdapter(inner: IncrSmallStrainModel, compliance: np.ndarray)` can be built, vectorized with `np.linalg.eigh`.

- It calls the **existing** Rust wrappers with `grad_del_u` $=\boldsymbol\varepsilon^e_\text{tr}$, passed as a symmetric 3×3 tensor, which `evaluate_model` passes through unchanged, and with `stress = 0`.
- This validates the solver, the tangent and all conventions with zero Rust changes.
- Afterwards it remains the test oracle for the Rust wrapper.

---

## 4. Caveats and limitations

- **Isotropy only.** Anisotropic elasticity or yield needs plastic spin or other kinematics.
- **Tensor-valued internal variables.** A back-stress for kinematic hardening is not covered by this algorithm as is. The existing `plastic_strain` history field (`RotatableVector`) becomes meaningless: it accumulates $\Delta\lambda\,\boldsymbol g$ in the spatial frame without rotation. Document this, or skip the update via the `return_map` refactor (§2).
- **Hencky elasticity.** See §6 for what this means in terms of the validity range.
- **Volumetric locking.** Isochoric plastic flow with P1/Q1 elements locks.
  - Use P2 initially.
  - F-bar fits this formulation naturally: replace $\boldsymbol F$ by $\bar{\boldsymbol F}$ before calling the model. But it needs element-wise averaging, which is awkward in dolfinx.
  - A mixed u–p formulation is a separate work package.
- **Time dependence.**
  - Viscoplastic models (Perzyna, etc.) carry over unchanged, since they are also isotropic return maps.
  - Viscoelastic models with stress history (spring–Kelvin/Maxwell) do **not** fit; they don't implement `HenckyElastic`.

---

## 5. Gradient-enhanced models

The existing small-strain mixed formulation (`IncrSmallStrainGradientProblem`) solves for $(\boldsymbol u, \bar\alpha)$ with an implicit-gradient (Helmholtz) equation. The coupling to the constitutive law goes through **scalars**:
- the local quantity $\alpha$, e.g. equivalent plastic strain or an equivalent strain,
- its nonlocal counterpart $\bar\alpha$.

Scalars don't change under rotation. So both the constitutive wrapper and the solver extend with little extra work. The main decisions are:
1. **How the local law is wrapped.** Incremental models (plasticity) use the multiplicative wrapper from §2. Total models (damage) use a simpler total-strain wrapper (§5.3).
2. **Which configuration the Helmholtz equation lives in** (§5.4).

### 5.1 Two kinds of local laws

| Kind | Examples | Stress | Wrapper |
|---|---|---|---|
| **Incremental** (return mapping) | `IsotropicGradientPlasticityModel3D`, Engelen-type overnonlocal plasticity | $\boldsymbol\tau = \mathbb C : \boldsymbol\varepsilon^e$ with $\boldsymbol\varepsilon^e$ from the exponential map | `Multiplicative<M>` (§5.2) |
| **Total** (stress depends only on the current strain) | Peerlings gradient damage, other elasticity-based damage | $\boldsymbol\tau = (1-\omega)\, \mathbb C : \boldsymbol\varepsilon$ | `TotalHencky<M>` (§5.3) |

### 5.2 Incremental gradient models (gradient plasticity)

`IsotropicGradientPlasticityModel3D` (branch `87-half-gradient-enahnced-plasticity`) has the same structure as the local model:
- It builds $\boldsymbol\sigma_\text{tr} = \mathbb C\, \Delta\boldsymbol\varepsilon + \boldsymbol\sigma_0$.
- It takes $\alpha_n$ from the `local_quantity` input and $\bar\alpha$ from `nonlocal_quantity`.
- It returns $\boldsymbol\sigma$, $\alpha_{n+1}$ and the four `NonlocalTangents`.

The `Multiplicative<M>` wrapper therefore applies directly:

- Call the model with `del_strain` $=\boldsymbol\varepsilon^e_\text{tr}$ and `stress = 0`.
- Pass `nonlocal_quantity` and `local_quantity` through untouched.
- Recover $\boldsymbol\varepsilon^e_{n+1}$ via `HenckyElastic::elastic_strain`. This model returns the effective stress `result.sigma` and stores $\omega$ only in its history, so $\mathbb C^{-1}\boldsymbol\tau$ is correct here.

The inner tangents are derivatives with respect to $\boldsymbol\varepsilon^e_\text{tr}$ and $\bar\alpha$. Using $\mathrm d\boldsymbol\varepsilon^e_\text{tr} = \tfrac12\mathbb L\mathbb B : \boldsymbol l$:

$$
\frac{\partial\boldsymbol\tau}{\partial\boldsymbol l} = \tfrac12\, \mathbb D_{\sigma\varepsilon}\, \mathbb L\, \mathbb B \;\;(6\times9),
\qquad
\frac{\partial\boldsymbol\tau}{\partial\bar\alpha} = \mathbb D_{\sigma\bar\alpha} \;\;(6),
$$

$$
\frac{\partial\alpha}{\partial\boldsymbol l} = \mathbb D_{\alpha\varepsilon}\, \tfrac12\, \mathbb L\, \mathbb B \;\;(1\times9),
\qquad
\frac{\partial\alpha}{\partial\bar\alpha} = \mathbb D_{\alpha\bar\alpha} \;\;(1).
$$

Here $\mathbb D_{\sigma\varepsilon}, \mathbb D_{\sigma\bar\alpha}, \mathbb D_{\alpha\varepsilon}, \mathbb D_{\alpha\bar\alpha}$ are `dsigma_deps`, `dsigma_dnonlocal`, `dlocal_deps`, `dlocal_dnonlocal` from the unchanged Rust model. $\mathbb L\mathbb B$ is already computed for the stress tangent, so the extra cost is one $1\times6 \cdot 6\times9$ product.

### 5.3 Total gradient models (Peerlings damage)

For elasticity-based damage, $\psi = (1-\omega)\,\psi_0(\boldsymbol\varepsilon)$ with the Hencky energy $\psi_0$. This is the *ideal* form for Hencky elasticity:

$$
\boldsymbol\tau = (1-\omega)\, \mathbb C : \boldsymbol\varepsilon,
\qquad
\boldsymbol\varepsilon = \tfrac12 \ln \boldsymbol b,
\qquad
\boldsymbol b = \boldsymbol F \boldsymbol F^T .
$$

**Consequences:**

- **No split of the deformation:** there's no $\boldsymbol F^p$, so multiplicative vs. additive doesn't matter. The model is a **total** formulation:
  - $\boldsymbol\varepsilon$ follows directly from $\boldsymbol F_{n+1}$,
  - no $\boldsymbol\varepsilon^e$ history, no "stress = 0" trick, no compliance,
  - the only history is $\omega$ (or $\kappa_\text{max}$).
- **Same result in either configuration:** for isotropic $\mathbb C$, the reference-configuration form $\boldsymbol T = (1-\omega)\mathbb C : \tfrac12\ln\boldsymbol C$ is the same law rotated, $\boldsymbol\tau = \boldsymbol R\boldsymbol T\boldsymbol R^T$.
- **Uniaxial stress stays linear:** $\tau_{11} = E \ln\lambda$ holds exactly, and the nominal stress is $P_{11} = \tau_{11}/\lambda$. The 1D Peerlings benchmark carries over cleanly.
- **Protection against volume collapse:** the volumetric energy $\tfrac12\kappa (\ln J)^2 \to \infty$ as $J \to 0$. As long as $\omega < 1$ (i.e. `omega_max < 1`), damaged material still resists collapsing to zero volume. Small-strain Peerlings has no such protection.

**Which strain drives $\omega$.** Use invariants of the **same Hencky strain that enters the stress**, i.e. functions of the principal log strains $\ln\lambda_i$:

$$
\alpha = \lVert\boldsymbol\varepsilon\rVert \;\;\text{(current choice)},
\qquad
\alpha = \sqrt{\textstyle\sum_i \langle \ln\lambda_i\rangle_+^2} \;\;\text{(Mazars)},
\qquad \text{or the modified von Mises equivalent strain.}
$$

Reasons:

1. **Doesn't depend on the configuration:** $\tfrac12\ln\boldsymbol C$ and $\tfrac12\ln\boldsymbol b$ have the same eigenvalues, so the equivalent strain is identical in the reference and the current configuration.
2. **Same small-strain limit:** $\boldsymbol\varepsilon\to\boldsymbol\varepsilon_\text{small}$, so `eps_0` keeps the meaning it has when calibrated from small-strain experiments.
3. **Symmetric in tension and compression:** $\ln\lambda$ treats stretching by $\lambda$ and compressing by $1/\lambda$ alike.
   - Green–Lagrange $\tfrac12(\lambda^2-1)$ overweights tension and is bounded by $-\tfrac12$ in compression.
   - Almansi does the opposite.

   Either would distort Mazars-type tension/compression definitions.
4. **The damage energy matches the stress:** with $\omega$ driven by an invariant of $\boldsymbol\varepsilon$ or by $\psi_0(\boldsymbol\varepsilon)$, the dissipation $\psi_0\,\dot\omega \ge 0$ holds trivially.

**Tangents.** In the implicit-gradient model, $\omega$ depends on $\bar\alpha$ (through its history maximum), not directly on $\boldsymbol\varepsilon$. So:

$$
\frac{\partial\boldsymbol\tau}{\partial\boldsymbol l} = (1-\omega)\, \tfrac12\, \mathbb C\, \mathbb L\, \mathbb B,
\qquad
\frac{\partial\boldsymbol\tau}{\partial\bar\alpha} = -\,\omega'(\bar\alpha)\; \mathbb C : \boldsymbol\varepsilon \quad(\text{on loading, else } 0),
\qquad
\frac{\partial\alpha}{\partial\bar\alpha} = 0 .
$$

For any isotropic scalar $\alpha = \varphi(\boldsymbol\varepsilon)$, the identity $\dot\alpha = \dfrac{\partial\varphi}{\partial\boldsymbol\varepsilon} : \boldsymbol d$ holds **exactly**, with $\boldsymbol d = \operatorname{sym}\boldsymbol l$. This is because $\partial\varphi/\partial\boldsymbol\varepsilon$ commutes with $\boldsymbol b$. Therefore

$$
\frac{\partial\alpha}{\partial\boldsymbol l} = \frac{\partial\varphi}{\partial\boldsymbol\varepsilon} \quad(\text{symmetric } 3\times3, \text{ flattened to } 9),
\qquad\text{e.g.}\quad
\frac{\partial\alpha}{\partial\boldsymbol l} = \frac{\boldsymbol\varepsilon}{\lVert\boldsymbol\varepsilon\rVert},
$$

with no $\mathbb L$ or $\mathbb B$ needed. Only the stress tangent needs the derivative of the logarithm.

**Implementation.**
- **Generic wrapper:** `TotalHencky<M>` in `finite_strain/total.rs`, for models that map the *total* strain to stress, $\boldsymbol\varepsilon\mapsto(\boldsymbol\tau, \mathbb D, \alpha, \dots)$. Per quadrature point:
  1. $\boldsymbol b = \boldsymbol F_1\boldsymbol F_1^T$
  2. one eigendecomposition $\to$ $\boldsymbol\varepsilon$, $\mathbb L$
  3. call the model
  4. assemble $\mathbb A = \tfrac12\mathbb D\mathbb L\mathbb B$ and $\partial\alpha/\partial\boldsymbol l$ as above
- **Rust Peerlings model:** add a small trait for total-strain models, `TotalStrainGradientModelFn::evaluate(strain, nonlocal, stress, local, tangents, history, params)`. The existing incremental `GradientConstitutiveModelFn` takes increments, so it doesn't fit.
- **Python:** the existing `PeerlingsGradientPerfectDamage` accumulates `total_strain` from increments. For the total wrapper it should receive $\boldsymbol\varepsilon$ directly (e.g. a `total_strain_input=True` flag), which makes the `total_strain` history redundant.
  - Alternative: the additive log-space wrapper from the companion plan runs the existing Python model **unchanged and exactly**, because increments of $\tfrac12\ln\boldsymbol C$ are additive.
- **Optional tension/compression split:** if needed, Hencky makes it easy. The principal log strains are already available from the eigendecomposition, and $\ln J = \operatorname{tr}\boldsymbol\varepsilon$ separates the volumetric part cleanly.

### 5.4 Configuration of the Helmholtz equation

Two options:

**(a) Reference configuration (default, recommended):**

$$
\bar\alpha - \ell^2\, \Delta_X \bar\alpha = \alpha \quad\text{in } \Omega_0 .
$$

- The weak form is exactly the current one, with `ufl.grad` meaning $\partial/\partial\boldsymbol X$.
- $\ell$ is a material length that stays constant, and the $\bar\alpha\bar\alpha$ block is unchanged.
- This is the common choice in finite-strain gradient plasticity and damage (e.g. Geers et al.).

**(b) Current configuration** (interaction length follows the deformed geometry, relevant in necking or shear bands):

$$
\int_{\Omega} \ell^2\, \nabla_x\bar\alpha\cdot\nabla_x\delta\bar\alpha \,\mathrm dv
+ \int_{\Omega} (\bar\alpha-\alpha)\,\delta\bar\alpha\,\mathrm dv
= \int_{\Omega_0} \Big( \ell^2\, \operatorname{Grad}\bar\alpha\cdot\boldsymbol C^{-1}\operatorname{Grad}\delta\bar\alpha + (\bar\alpha-\alpha)\,\delta\bar\alpha \Big)\, J\, \mathrm dX .
$$

- This adds a $\partial/\partial\boldsymbol u$ contribution through $\boldsymbol C^{-1}$ and $J$.
- These terms depend on $\boldsymbol u$ only through UFL expressions, so `ufl.derivative` with respect to $\boldsymbol u$ gives that part exactly.
- `ufl.derivative` treats the quadrature function $\alpha$ as constant, so the dependence of $\alpha$ on $\boldsymbol u$ must still be added by hand (block $\bar\alpha u$ below, multiplied by $J$).
- Offer it as a boolean option on the solver.

### 5.5 Mixed solver `IncrFiniteStrainGradientProblem`

This is the same structure as `IncrSmallStrainGradientProblem`, with the displacement part replaced by the finite-strain forms. With option (a):

$$
R_u = \int_{\Omega_0} \boldsymbol\tau : \boldsymbol l(\boldsymbol v)\,\mathrm dX - f_\text{ext}(\boldsymbol v),
\qquad
R_{\bar\alpha} = \int_{\Omega_0} \Big( \ell^2 \operatorname{Grad}\bar\alpha\cdot\operatorname{Grad}\delta\bar\alpha + (\bar\alpha - \alpha)\,\delta\bar\alpha \Big)\mathrm dX .
$$

Jacobian blocks:

$$
\begin{aligned}
K_{uu} &= \int_{\Omega_0} \Big(\frac{\partial\boldsymbol\tau}{\partial\boldsymbol l} : \boldsymbol l(\Delta\boldsymbol u)\Big) : \boldsymbol l(\boldsymbol v)\,\mathrm dX
 - \int_{\Omega_0} \boldsymbol\tau : \big(\boldsymbol l(\boldsymbol v)\,\boldsymbol l(\Delta\boldsymbol u)\big)\,\mathrm dX, \\
K_{u\bar\alpha} &= \int_{\Omega_0} \Big(\frac{\partial\boldsymbol\tau}{\partial\bar\alpha}\,\Delta\bar\alpha\Big) : \boldsymbol l(\boldsymbol v)\,\mathrm dX, \\
K_{\bar\alpha u} &= -\int_{\Omega_0} \delta\bar\alpha\; \frac{\partial\alpha}{\partial\boldsymbol l} : \boldsymbol l(\Delta\boldsymbol u)\,\mathrm dX, \\
K_{\bar\alpha\bar\alpha} &= \int_{\Omega_0} \Big( \ell^2 \operatorname{Grad}\Delta\bar\alpha\cdot\operatorname{Grad}\delta\bar\alpha + \Big(1 - \frac{\partial\alpha}{\partial\bar\alpha}\Big)\Delta\bar\alpha\,\delta\bar\alpha \Big)\mathrm dX .
\end{aligned}
$$

In UFL (with `T` = Mandel→tensor, `flat` = 3×3→9):

```python
R  = inner(tau_t, lx(v))*dX - f_ext \
   + l**2*inner(grad(nl), grad(dnl))*dX + (nl - alpha)*dnl*dX

J  = inner(T(dot(A, flat(lx(du)))), lx(v))*dX - inner(tau_t, lx(v)*lx(du))*dX     # uu
   + inner(T(dtau_dnl), lx(v)) * nl_trial * dX                                   # u-ᾱ
   - dnl * inner(dalpha_dl, flat(lx(du))) * dX                                    # ᾱ-u
   + l**2*inner(grad(nl_trial), grad(dnl))*dX + (1 - dalpha_dnl)*nl_trial*dnl*dX  # ᾱ-ᾱ
```

**Interfaces and infrastructure:**

- **Python ABC** `IncrFiniteStrainGradientModel`: `evaluate(t, dt, grad_u, grad_del_u, nonlocal_quantity, stress, local_quantity, tangents, history)`.
  - `stress` is $\boldsymbol\tau$.
  - `tangents` is a `FiniteStrainNonlocalTangents` with shapes $6\times9$, $6$, $9$, $1$.
- **Rust:** `FiniteStrainNonlocalTangents { dtau_dl: SMatrix<6,9>, dtau_dnonlocal: SVector<6>, dlocal_dl: RowSVector<9>, dlocal_dnonlocal: f64 }`, plus a trait `FiniteStrainGradientConstitutiveModelFn` with blanket implementations:
  - `Multiplicative<M>` for `M: GradientConstitutiveModelFn<6,…> + HenckyElastic`,
  - `TotalHencky<M>` for `M: TotalStrainGradientModelFn<6,…>`.
- **Spaces:** `GradientElements` gets finite-strain variants of the `dsigma_deps` ($6\times9$) and `dlocal_deps` ($9$) spaces.
- **Kinematics:**
  - `IncrementalGradientSolution` gets a total-gradient `Expression` next to `_del_grad_u_expr`.
  - `GradientLawOnSubMesh` gets a `FiniteStrainGradientLawOnSubMesh` subclass with the extra `grad_u` function.
- **Bindings:** an `implement_finite_strain_gradient_python_model!` macro, one line per model.

### 5.6 Practical points

- **Element choice:** P2 displacement with a P1 nonlocal field. Locking from isochoric plastic flow combined with softening is worse than either effect alone.
- **Inverted elements:** heavily softened elements can collapse. The wrapper should check $\det\boldsymbol F > 0$ and report failure instead of panicking, so the time step can be cut.
- **Tangent transposes:** the small-strain Jacobian uses `dsigma_deps` transposed (the `transpose_mut()` in Rust), while `dsigma_dnonlocal` and `dlocal_deps` are used untransposed. Fix one convention for the new $6\times9$ and $9$ blocks and document it.
- **Overnonlocal formulation:** the parameter $m$ mixes $\alpha$ and $\bar\alpha$. These are scalars, so it is unaffected by the kinematics.

---

## 6. Validity range of Hencky elasticity: elastic vs. total strains

The limits of the quadratic Hencky energy apply to **whatever strain enters $\psi$**:

- **Uniaxial tension:** the nominal stress $P = E\ln\lambda/\lambda$ has its maximum at $\lambda = e$, i.e. about 170% strain.
- **3D:** the requirement that the elastic law stays stable for all deformation modes (rank-one convexity / ellipticity) fails earlier than that, depending on the combination of principal stretches.
- **Reference for exact bounds:** Bruhns, Xiao & Meyers (2001). In practice, Hencky elasticity is well-behaved for moderately large strains.

What this means for each model kind:

| Model | Strain in $\psi$ | Consequence |
|---|---|---|
| Multiplicative plasticity (local or gradient) | elastic $\boldsymbol\varepsilon^e$, bounded by yield to about $\sigma_y/E \sim 10^{-3}$ | **Irrelevant**, even for total strains of 100% or more (necking). This is the main reason Hencky elasticity combined with multiplicative plasticity is so widely used. |
| Total damage models (Peerlings) | **total** $\boldsymbol\varepsilon$ | Only matters inside highly damaged bands, where the stiffness is already reduced to $(1-\omega_\text{max})\mathbb C$. Softening makes the local problem ill-posed long before Hencky's own limit; the gradient regularization fixes that via $\bar\alpha$ but does *not* regularize an instability of $\psi_0$ itself. |

**Measures:**
- Keep `omega_max < 1`.
- Optionally warn when $\max_i \lvert\ln\lambda_i\rvert$ exceeds a threshold, and check $\det\boldsymbol F > 0$.
- If large elastic strains are really needed, swap $\psi_0$ for an energy that stays well-behaved, such as Neo-Hooke or the exponentiated Hencky energy of Neff et al. That turns $\mathbb C$ into a strain-dependent tangent; the rest of the setup is unchanged.

---

## 7. Test plan

| Level | Test |
|---|---|
| Rust unit: spectral | $\exp(\ln\boldsymbol b) = \boldsymbol b$. $\mathbb L$ vs. finite differences, including $\boldsymbol b = \boldsymbol I$ and $\boldsymbol b = \operatorname{diag}(a,a,c)$. |
| Rust unit: tangent | $\mathbb A$ vs. central FD of $\boldsymbol\tau(\boldsymbol F_1 + h\operatorname{Grad}\Delta\boldsymbol u)$, for elastic and plastic steps, associated and non-associated DP, random $\boldsymbol F$. |
| Rust unit: gradient tangents | All four `FiniteStrainNonlocalTangents` vs. FD (perturb $\boldsymbol F_1$ and $\bar\alpha$). Check the identity $\partial\alpha/\partial\boldsymbol l = \partial\varphi/\partial\boldsymbol\varepsilon$ for total models. |
| Solver: Taylor test | Generic helper: $\lVert R(\boldsymbol u + h\Delta\boldsymbol u) - R(\boldsymbol u) - h\, J\,\Delta\boldsymbol u\rVert = \mathcal O(h^2)$, for the local **and** the mixed $(\boldsymbol u,\bar\alpha)$ problem. This catches transpose, `flatten`-ordering and geometric-term errors. |
| Small-strain limit | Tiny displacements reproduce `IncrSmallStrainProblem` / `IncrSmallStrainGradientProblem`. |
| Objectivity | Stress-free rigid rotation gives $\boldsymbol\tau = \boldsymbol 0$. Stretch, then rotate about a **non-z axis**: $\boldsymbol\tau \mapsto \boldsymbol R\boldsymbol\tau\boldsymbol R^T$, and $\alpha$, $\bar\alpha$, $\omega$ stay unchanged. |
| Analytic | Elastic Hencky uniaxial stress: $\tau_{11} = E\ln\lambda$. 1D Peerlings bar with Hencky strain, compared against the small-strain analytic solution at small loads. |
| Cross-check with log-space plan | Coaxial/proportional loading gives identical results. Peerlings damage gives identical results for **all** loadings (no plastic split). Non-proportional plasticity gives small, quantified differences. |
| Newton | Quadratic convergence in plastic and damaging steps. |
| Benchmarks | Necking of a circular bar (Simo 1988/1992) for local plasticity. Necking or shear band with gradient plasticity: the band width should converge with mesh refinement. |
| MPI | Add to the `test_solver_mpi.py` patterns. |

---

## 8. Work order

0. **Prep** (shared with the other plan):
   - Fix the Mandel ordering in `corotational_lawonsubmesh.py`.
   - Fix the `nabla_grad`/`grad` and tangent-layout conventions.
   - Add the `mandel_to_tensor` and `flatten` UFL helpers.
1. **Local solver + interface + numpy wrapper:** `IncrFiniteStrainModel`, `IncrFiniteStrainProblem`, `MultiplicativeAdapter` (Python). Taylor, small-strain-limit and objectivity tests.
   - At this point every existing Rust plasticity model already runs with multiplicative kinematics.
2. **Rust local wrapper:**
   - `HenckyElastic` and its implementations
   - `finite_strain/{spectral, tangent, multiplicative}.rs`
   - `evaluate_finite_strain_model`
   - bindings macro and Python wrappers
   - cross-check against the numpy wrapper
3. **Gradient-enhanced, reference-configuration Helmholtz:**
   - `IncrFiniteStrainGradientModel`, `IncrFiniteStrainGradientProblem`.
   - A numpy `TotalHencky` wrapper for Peerlings, so the 1D/2D Peerlings tests can be repeated at finite strain.
   - A numpy `Multiplicative` wrapper for gradient plasticity, once `general_gradient_enhanced.rs` is merged.
   - Mixed Taylor test.
4. **Rust gradient wrappers:** `FiniteStrainNonlocalTangents`, `TotalHencky<M>`, `Multiplicative<M>` for `GradientConstitutiveModelFn`, the bindings macro.
5. **Cleanup in `general.rs` / `general_gradient_enhanced.rs`:** extract `return_map(...)` so the wrappers no longer rely on `sigma_0 = 0` or the `plastic_strain` side effect.
6. **Extensions:**
   - Current-configuration Helmholtz option (§5.4b).
   - Plane strain: $F_{33} = 1$, use the 3D model, project $\boldsymbol\tau$ and $\mathbb A$ to 4 / $4\times4$ components.
   - Explicit dynamics: residual only, `tangent=None`.
   - Tension/compression split for damage.
7. **Docs:** a formulation page, the necking example, and a finite-strain Peerlings example.

---

## 9. Comparison with the additive log-strain-space plan

| | Multiplicative (this plan) | Additive log-space ([FINITE_STRAIN_PLAN.md](FINITE_STRAIN_PLAN.md)) |
|---|---|---|
| Kinematics | exact $\boldsymbol F = \boldsymbol F^e\boldsymbol F^p$ (zero plastic spin) | additive $\boldsymbol E_{\log} = \boldsymbol E^e + \boldsymbol E^p$; equals multiplicative only for coaxial loading |
| Model class | isotropic only (opt-in via `HenckyElastic`) | any additive small-strain model, including anisotropic and viscoelastic |
| Damage (Peerlings) | exact via `TotalHencky` | exact, existing Python model runs unchanged |
| Rust model changes | none (one small trait impl per model family) | none |
| Log derivatives needed | first only | first and second |
| Stored state | $\boldsymbol\varepsilon^e$ (spatial, 6) + inner scalars | $\boldsymbol T_n$ (6) + inner history (in the reference configuration) |
| Tensor-valued inner history | not meaningful | meaningful (reference configuration) |
| Stress/tangent from model | $\boldsymbol\tau$, $\partial\boldsymbol\tau/\partial\boldsymbol l$ ($6\times9$) | $\boldsymbol S$, $\partial\boldsymbol S/\partial\boldsymbol E$ ($6\times6$) |
| Literature benchmarks | directly comparable (Simo) | approximately comparable |

**Shared solver.** Both can share one solver. If the $(\boldsymbol\tau, \partial\boldsymbol\tau/\partial\boldsymbol l)$ interface of §3.1 is adopted as the single finite-strain interface, the log-space wrapper becomes just another model behind it: it converts $\boldsymbol S$ to $\boldsymbol\tau = \boldsymbol F\boldsymbol S\boldsymbol F^T$ and pushes its tangent forward. The same holds for the gradient-enhanced variants.

## References

- J.C. Simo (1992). *Algorithms for static and dynamic multiplicative plasticity that preserve the classical return mapping schemes of the infinitesimal theory.* CMAME 99, 61–112.
- A.L. Eterovic, K.-J. Bathe (1990). *A hyperelastic-based large strain elasto-plastic constitutive formulation with combined isotropic-kinematic hardening using the logarithmic stress and strain measures.* IJNME 30.
- E.A. de Souza Neto, D. Perić, D.R.J. Owen (2008). *Computational Methods for Plasticity*, Wiley, Ch. 14.
- C. Miehe, M. Lambrecht (2001). *Algorithms for computation of stresses and elasticity moduli in terms of Seth–Hill's family of generalized strain tensors.* Commun. Numer. Meth. Engng 17.
- R.H.J. Peerlings, R. de Borst, W.A.M. Brekelmans, J.H.P. de Vree (1996). *Gradient enhanced damage for quasi-brittle materials.* IJNME 39.
- M.G.D. Geers, R.L.J.M. Ubachs, R.A.B. Engelen (2003). *Strongly non-local gradient-enhanced finite strain elastoplasticity.* IJNME 56.
- O.T. Bruhns, H. Xiao, A. Meyers (2001). *Constitutive inequalities for an isotropic elastic strain-energy function based on Hencky's logarithmic strain tensor.* Proc. R. Soc. A 457.
- P. Neff, I.-D. Ghiba, J. Lankeit (2015). *The exponentiated Hencky-logarithmic strain energy. Part I: constitutive issues and rank-one convexity.* J. Elasticity 121.
