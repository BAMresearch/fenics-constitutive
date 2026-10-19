# Plan: Finite-strain models in fenics-constitutive

**Goal:** support finite-strain (large deformation) constitutive models while reusing the existing
Rust plasticity implementations (`IsotropicPlasticityModel3D<…>`, `MisesPlasticity3D`,
`DruckerPrager*`, …) **without modifying them**.

**Recommendation in one sentence:** add a Total-Lagrangian finite-strain solver that works with
(2nd Piola–Kirchhoff stress `S`, Green–Lagrange strain `E`) in Mandel notation, plus a generic
**logarithmic-strain-space adaptor** (Miehe, Apel & Lambrecht 2002) that wraps *any* existing
small-strain `ConstitutiveModelFn<6, …>` / `IncrSmallStrainModel` and turns it into a finite-strain
model.

---

## 1. Current state (what the repo gives us)

| Component | Relevant property |
|---|---|
| `ConstitutiveModelFn::evaluate(t, dt, del_strain, stress, tangent, history, params)` | Purely **incremental and additive**: `σ_tr = σ_n + C:Δε`, return mapping, algorithmic tangent `∂σ_{n+1}/∂Δε`. Knows nothing about kinematics. |
| `evaluate_model` (`comfe-rs/src/interfaces.rs`) | Converts the raw 3×3 `grad_del_u` to Mandel by symmetrizing (`nonsymmetric_tensor_to_mandel`). Passing a *symmetric* 3×3 tensor therefore passes it through unchanged. |
| `ElementSpaces` | Quadrature spaces for Mandel stress (6), tangent (6×6), and displacement gradient (3×3). |
| `History`, `LawOnSubMesh`, `IncrementalDisplacement`, `IncrementalStress` | Model-agnostic bookkeeping that can be reused almost unchanged. |
| `CorotationalIncrSmallStrainProblem` | Hypoelastic updated-Lagrangian approach: mesh moved to midpoint, Cauchy stress half-rotated before and after the law. |

### Limitations of the existing corotational approach

The corotational approach is useful, but it has limits that a TL/log-strain formulation avoids:

1. **Hypoelasticity**: closed elastic cycles can dissipate or generate energy, and a Jaumann-type rate oscillates in simple shear. (The incremental Hughes–Winget rotation used here helps but does not remove the issue.)
2. **Inconsistent tangent**: no geometric stiffness and no linearization of the rotation, so Newton converges at best linearly at large strains.
3. **History is not rotated**: fields declared `QDim::RotatableVector`, e.g. `plastic_strain`, stay in the old frame.
4. **Bug**: `CorotationalLawOnSubMesh._stress_rotate` maps Mandel index 4 to `(1,2)` and index 5 to `(0,2)`. Rust (`nonsymmetric_tensor_to_mandel`) and `ufl_mandel_strain` use **4 = (0,2), 5 = (1,2)**. The round trip is self-consistent, but the rotation is applied to a tensor with xz and yz swapped, which is wrong for any rotation not about the z-axis. `test_corotational_solver.py` only rotates about z, so it doesn't catch this. **Fix this independently of the rest of the plan.**

---

## 2. Options considered

| | A: Improve corotational (hypoelastic) | **B: Log-strain-space adaptor (recommended)** | C: Rewrite models multiplicatively (`F = Fe·Fp`, Simo-type) |
|---|---|---|---|
| Rust model changes | none | **none** | every model needs a new return map in `b_e`/Kirchhoff space |
| Elasticity | hypoelastic | hyperelastic (Hencky) | hyperelastic |
| Consistent tangent / quadratic Newton | hard | **yes (exact chain rule)** | yes |
| History handling | must rotate tensorial history | **Lagrangian, no rotation needed** | spatial, needs push-forward |
| Anisotropic / non-associated models | ok | ok (works for any additive model) | isotropy usually required |
| Mesh movement | yes (isoparametric only) | **no (TL)** | depends |
| Effort | small | medium | large |

**Why B fits this repository well:** the Rust plasticity models are already written as additive
small-strain return maps, and that is exactly what the log-strain-space formulation needs. The
whole finite-strain part becomes a geometric pre- and post-processor around an unmodified model.

Option A is still worth keeping for explicit dynamics and for comparison, once the bug above is fixed.

---

## 3. The log-strain-space formulation (math)

Per quadrature point, at step `n → n+1`:

1. **Kinematics.** `F = I + Grad u`. `C = Fᵀ F`.
   - ⚠️ dolfinx/UFL convention: `ufl.nabla_grad(u)[i,j] = ∂u_j/∂X_i = (Grad u)ᵀ`. The existing `IncrementalDisplacement._expr` uses `nabla_grad`, so `F = I + nabla_grad(u)ᵀ`. Pick one convention and test it explicitly with a pure shear field, where the transpose matters.
2. **Lagrangian log strain.** `E_log = ½ ln C = Σ_a ½ ln(λ_a) N_a ⊗ N_a`, where (`λ_a`, `N_a`) are the eigenpairs of `C`.
3. **Call the unchanged small-strain model:**
   - `Δε := E_log,n+1 − E_log,n` (Mandel 6)
   - `stress_in := T_n` (the log-space stress, stored as history)
   - returns `T_{n+1}` and `𝔻 = ∂T/∂E_log` (6×6, algorithmic)
   - The inner model's history (`alpha`, `plastic_strain`, …) lives in the Lagrangian log space. **No rotation is ever needed.**
4. **Map back to the TL pair (S, E_GL)** with `E_GL = ½(C − I)`:
   - `ℙ := ∂E_log/∂E_GL = 2 ∂E_log/∂C` (6×6 in Mandel, symmetric)
   - `S = ℙᵀ T` (in Mandel this is a plain matrix–vector product because the basis is orthonormal)
   - `∂S/∂E_GL = ℙᵀ 𝔻 ℙ + T : 𝕃`, where `𝕃 = ∂²E_log/∂E_GL²`, so `T:𝕃` is a 6×6 matrix.

**Derivatives of the matrix logarithm (Daleckii–Krein).** With `C = Q Λ Qᵀ`:

- First derivative: `dE_log[H] = Q (Γ ∘ (Qᵀ (2H) Q)) Qᵀ`, where
  - `Γ_ab = (f(λ_a) − f(λ_b))/(λ_a − λ_b)` for `a ≠ b`,
  - `Γ_aa = f'(λ_a)`,
  - `f(λ) = ½ ln λ`.
  - Build `ℙ` by applying this to the 6 Mandel basis tensors.
- Second derivative: the analogous formula with second-order divided differences (Miehe & Lambrecht 2001, *Algorithms for computation of stresses and elasticity moduli in terms of Seth–Hill's family of generalized strain tensors*).
- **Pragmatic first version:** compute `T:𝕃` by central finite differences of `E ↦ ℙ(E)ᵀ T` (6 perturbations, i.e. 12 eigendecompositions of a 3×3 matrix). Replace it with the analytic form later and keep the FD version as a test oracle.
- **Repeated eigenvalues are the common case:** at `F = I` all three are equal. Use the limits `Γ_ab → f'(λ)` and `→ f''(λ)/2` whenever `|λ_a − λ_b| < tol·max(λ)`. At `C = I` the result must be `ℙ = I`.

**What this formulation is:**

- Elasticity becomes Hencky hyperelasticity (`T = C_el : E_log,e`). This is very good for moderate elastic strains, which covers metals, concrete, and geomaterials.
- The additive split `E_log = E_e + E_p` coincides with multiplicative plasticity in coaxial cases. In general it is a well-established approximation (Miehe et al. 2002; Papadopoulos & Lu 1998/2001). The differences are small when elastic strains are small.
- Non-associated flow (Drucker–Prager with `b_flow ≠ b`) and anisotropy carry over directly.

---

## 4. Software design

### 4.1 Python interface (`models/interfaces.py`)

Add a third model interface next to the two existing ones:

```python
class IncrFiniteStrainModel(ABC):
    @abstractmethod
    def evaluate(
        self,
        t: float,
        del_t: float,
        grad_u: np.ndarray,          # total displacement gradient at t_{n+1} (raw, non-symmetric, gdim×gdim per QP)
        grad_del_u: np.ndarray,      # increment, same layout as for small strain
        stress: np.ndarray,          # in: S_n, out: S_{n+1}  (2nd Piola–Kirchhoff, Mandel)
        tangent: np.ndarray | None,  # out: dS/dE_GL (Mandel 6×6), None => skip
        history: dict[str, np.ndarray] | None,
    ) -> None: ...
    constraint / history_dim  # as in IncrSmallStrainModel
```

- Keep `S` and a 6×6 tangent so that `ElementSpaces`, `History`, and `IncrementalStress` can be reused as-is.
- `F_n` can be reconstructed as `I + (grad_u − grad_del_u)ᵀ`, so nothing extra has to be stored.

### 4.2 Total-Lagrangian solver (`solver/_finite_strain_solver.py`)

`IncrFiniteStrainProblem(NonlinearProblem)` mirrors `IncrSmallStrainProblem`:

```python
F  = ufl.Identity(d) + ufl.grad(u)
def dE(v):  # variation of Green–Lagrange strain, Mandel vector (same ordering as ufl_mandel_strain)
    return mandel(ufl.sym(F.T * ufl.grad(v)))

R  = ufl.inner(S, dE(v)) * dxm - sum(external_forces)
J  = ufl.inner(ufl.dot(D, dE(du)), dE(v)) * dxm \
   + ufl.inner(mandel_to_tensor(S), ufl.grad(du).T * ufl.grad(v)) * dxm   # geometric stiffness
```

- UFL handles the geometric stiffness exactly, so only the material part comes from the model.
- **Tangent layout:** the existing small-strain Jacobian is `ε(du)·(D·ε(v))`, which uses `Dᵀ`. That is why `general.rs` transposes the tangent (`// TODO: move the transpose to the python bindings`). Choose one convention for the new solver, document it, and make the adaptor in §4.3/4.4 respect it. This matters for non-associated Drucker–Prager, where `D` is non-symmetric.
- The mesh is never moved, so there's no isoparametric restriction and no `MeshUpdater`.
- Reuse the existing pieces:
  - `IncrementalDisplacement`: add a second `Expression` for the total gradient.
  - `LawOnSubMesh`: subclass it as `FiniteStrainLawOnSubMesh` with one extra `grad_u` quadrature function.
  - `History` and `build_subspace_map`: unchanged.
- External forces are dead loads in the reference configuration. Follower loads (pressure) would need an extra linearization term; leave that for later.
- Add a postprocessing helper for Cauchy stress, `σ = J⁻¹ F S Fᵀ`, and Kirchhoff stress for sensors.

### 4.3 Phase-1 adaptor in Python (zero Rust changes)

`LogStrainAdapter(inner: IncrSmallStrainModel) -> IncrFiniteStrainModel` in numpy, vectorized with
`np.linalg.eigh` on `(n_qp, 3, 3)` arrays:

- `history_dim = inner.history_dim ∪ {"log_stress": 6}`, optionally plus `{"log_strain": 6}` to save one eigendecomposition.
- It calls `inner.evaluate(t, dt, ΔE_log_as_3x3, T, D, inner_history)`.

**Why this already works for Rust models:** `evaluate_model` symmetrizes `grad_del_u`, so passing the symmetric 3×3 `ΔE_log` reproduces the Mandel `Δε` exactly. Every existing Rust plasticity model is therefore usable at finite strain from day one, and so are the Python models (viscoelasticity, Python Mises).

This adaptor also serves as the **reference implementation** for testing the Rust version.

### 4.4 Phase-2 adaptor in Rust (performance)

In `comfe-rs/src/finite_strain/`:

- `spectral.rs`: symmetric 3×3 eigendecomposition (`nalgebra::SymmetricEigen`), plus `log`/`exp` and first/second divided-difference derivatives, with robust handling of repeated eigenvalues.
- `interfaces.rs`: a new trait plus an evaluation loop over quadrature points.

  ```rust
  pub trait FiniteStrainConstitutiveModelFn<const N_HISTORY, const HISTORY, const N_PARAMETERS, const PARAMETERS> {
      type History; type Parameters;
      fn evaluate(time, del_time, f_0: &[[f64;3];3], f_1: &[[f64;3];3],
                  stress_pk2: &mut [f64;6], tangent: Option<&mut [[f64;6];6]>,
                  history: &mut [f64; HISTORY], parameters: &[f64; PARAMETERS]);
  }
  pub fn evaluate_finite_strain_model<…>(…)  // analogous to evaluate_model
  ```

- `log_strain.rs`: `pub struct LogStrainSpace<M>(PhantomData<M>)`, with a blanket implementation for every `M: ConstitutiveModelFn<6, NH, H, NP, P>`.
  - ⚠️ `H + 6` in a const generic needs the unstable `generic_const_exprs`. Avoid it by keeping `T_n` in a **separate history entry** (`"log_stress"`) instead of concatenating it into `"history"`. The bindings already receive history as a `HashMap`.
  - Mind the tangent memory layout: nalgebra's `.data.0` is column-major, and `general.rs` transposes before writing. Wrap the inner tangent with the matching `from_row_slice`/`from_column_slice`, and cover it with a non-symmetric Drucker–Prager test.
- `bindings/src/lib.rs`: add an `implement_log_strain_python_model!(m, PyLogStrainIsotropicMises3D, IsotropicPlasticityModel3D<4,4,IsotropicMises3D>, StressStrainConstraint::FULL)` macro. With it, **one line per model** exposes the finite-strain version.
- `models/rust_models.py`: add a `fenics_constitutive_finite_strain_wrapper` decorator, the analogue of `fenics_constitutive_wrapper` asserting against `IncrFiniteStrainModel`.

### 4.5 Later extensions

- **Plane strain:** embed `F` in 3D with `F₃₃ = 1`, run the 3D Rust model, and project `S` and `D` from 6 to the 4-component Mandel layout. As a side effect, this makes the 3D Rust models usable in 2D.
- **Gradient-enhanced models:** use the same adaptor around `GradientConstitutiveModelFn`. The extra tangent blocks transform as:
  - `dS/dnl = ℙᵀ ∂T/∂nl`
  - `dlocal/dE_GL = ∂local/∂E_log · ℙ`
  - `dlocal/dnl` is unchanged.

  Then add a finite-strain variant of `IncrSmallStrainGradientProblem` with the same form changes as in §4.2.
- **Explicit dynamics (`CDMSolver`):** this only needs the residual, `∫ S : δE dX`, with `tangent=None`. That is straightforward once §4.2 exists.
- **Volumetric locking:** isochoric plastic flow with P1/Q1 elements locks. Use P2 initially. F-bar or a mixed u–p formulation is a separate work package.

---

## 5. Test plan

| Level | Test |
|---|---|
| Rust unit | Check `exp(log(C)) = C`. Check `ℙ` against finite differences, including repeated eigenvalues (`C = I`, `C = diag(a,a,b)`). Check the full `∂S/∂E_GL` against FD for elastic and plastic steps (extend the existing tangent tests from #95). |
| Python adaptor vs Rust adaptor | Identical `S`, `D`, and history for random `F` and all wrapped models. |
| Small-strain limit | For displacements around 1e-6, results match `IncrSmallStrainProblem` with the same inner model. |
| Objectivity | Apply a rigid rotation of a stress-free body and check `S = 0`. Stretch, then rotate, and check that `S` stays constant while the Cauchy stress rotates. Use rotation about a **non-z axis**. |
| Analytic elastic | Uniaxial and biaxial stretch of a Hencky material, e.g. `T = 2μ dev E_log + κ tr E_log I`, with an analytic `S(λ)`. |
| Simple shear | Large-shear Mises response: no stress oscillations, unlike a Jaumann rate. Compare with the corotational solver to show the difference. |
| Newton convergence | Quadratic convergence rate in plastic steps, asserted on the residual history. |
| Benchmark | Necking of a circular bar (Simo 1988 / Simo & Armero) to compare the load–displacement curve with literature. |
| MPI | Add the new solver to `test_solver_mpi.py` patterns. |

---

## 6. Suggested order of work

0. **Prep**
   - Fix the Mandel ordering in `corotational_lawonsubmesh.py` and add a rotation-about-x test.
   - Document the tangent layout and transpose convention, and the `nabla_grad` vs `grad` convention.
1. **Python vertical slice:** the `IncrFiniteStrainModel` ABC, `IncrFiniteStrainProblem`, the numpy `LogStrainAdapter` (FD for `T:𝕃`), and tests for the small-strain limit, objectivity, analytic elastic response, and Newton convergence.
   - At this point all Rust plasticity models already run at finite strain.
2. **Rust adaptor:** `spectral.rs`, analytic second derivative, `LogStrainSpace<M>`, the bindings macro, the Python wrapper, and cross-checks against the Phase-1 adaptor.
3. **Extensions:** plane strain, then the gradient-enhanced finite-strain solver, then CDM.
4. **Docs:** a `docs/` page on the formulation and a necking example.

## 7. Files touched (summary)

- **New:**
  - `src/fenics_constitutive/models/log_strain_adapter.py`
  - `src/fenics_constitutive/solver/_finite_strain_solver.py`
  - `comfe-rs/src/finite_strain/{mod,spectral,interfaces,log_strain}.rs`
  - tests under `tests/models/` and `tests/solver/`
- **Modified:**
  - `models/interfaces.py`: new ABC
  - `solver/_incrementalunknowns.py`: total-gradient expression
  - `solver/_lawonsubmesh.py`: finite-strain subclass
  - `solver/__init__.py`
  - `solver/utils.py`: `mandel_to_tensor` UFL helper
  - `models/rust_models.py`
  - `bindings/src/lib.rs`
  - `comfe-rs/src/lib.rs`
- **Unchanged:** all files in `comfe-rs/src/plasticity/`, `mises_plasticity.rs`, `linear_elasticity.rs`.

## References

- C. Miehe, N. Apel, M. Lambrecht (2002). *Anisotropic additive plasticity in the logarithmic strain space: modular kinematic formulation and implementation based on incremental minimization principles for standard materials.* CMAME 191.
- C. Miehe, M. Lambrecht (2001). *Algorithms for computation of stresses and elasticity moduli in terms of Seth–Hill's family of generalized strain tensors.* Commun. Numer. Meth. Engng 17.
- P. Papadopoulos, J. Lu (1998, 2001). *A general framework for the numerical solution of problems in finite elasto-plasticity.* CMAME 159 / 190.
- J.C. Simo (1988, 1992). Multiplicative finite-strain plasticity and the necking benchmark. CMAME 66 / 99.
