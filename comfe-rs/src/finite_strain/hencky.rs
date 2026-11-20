//! Hencky elasto-plasticity with the multiplicative split $\mathbf F = \mathbf F^e\mathbf F^p$ and
//! the exponential-map return mapping.
//!
//! For isotropic models, the return mapping in logarithmic strain space has the same algebraic
//! form as the small-strain backward-Euler return mapping, with the Kirchhoff stress
//! $\boldsymbol\tau$ in place of $\boldsymbol\sigma$ and the trial stress
//! $\boldsymbol\tau^{tr} = \mathbb C\thinspace \tfrac12\ln\mathbf b^e_{tr}$,
//! $\mathbf b^e_{tr} = \mathbf F\mathbf C_{p,n}^{-1}\mathbf F^T$, where $\mathbf C_{p,n}^{-1}$ is the
//! plastic history (see [below](#plastic-history)). Every model implementing [`Plasticity`] is
//! therefore reused unchanged through [`NewtonSolverStrain`]. The yield function is evaluated in
//! Kirchhoff stress, or optionally in Cauchy stress (see [`HenckyPlasticity3D`]).
//!
//! The models return $\boldsymbol\tau$ and the tangent $\mathbb a = \partial\boldsymbol\tau/
//! \partial\mathbf d$, see [`FiniteStrainConstitutiveModelFn`].
//!
//! # Plastic history
//!
//! The plastic deformation is stored as the inverse plastic right Cauchy–Green tensor
//!
//! $$\mathbf C_p^{-1} = \big(\mathbf F^{pT}\mathbf F^p\big)^{-1},$$
//!
//! a symmetric tensor in the reference configuration. It takes the role of the plastic strain of
//! the small-strain models: just as $\boldsymbol\varepsilon_e = \boldsymbol\varepsilon -
//! \boldsymbol\varepsilon_p$, it removes the plastic part from the total deformation,
//!
//! $$\mathbf b^e = \mathbf F^e\mathbf F^{eT} = \mathbf F\thinspace\mathbf C_p^{-1}\mathbf F^T .$$
//!
//! - **Why not $\mathbf F^p$:** the split $\mathbf F = \mathbf F^e\mathbf F^p$ is only unique up to a
//!   rotation $\mathbf Q$ of the intermediate configuration, $\mathbf F^e\mathbf F^p =
//!   (\mathbf F^e\mathbf Q)(\mathbf Q^T\mathbf F^p)$. $\mathbf C_p^{-1}$ and $\mathbf b^e$ do not depend on
//!   $\mathbf Q$, and as a symmetric tensor $\mathbf C_p^{-1}$ needs 6 values instead of 9.
//! - **Never rotate it:** superposed rigid rotations $\mathbf F \to \mathbf R\mathbf F$ of the current
//!   configuration leave $\mathbf C_p^{-1}$ unchanged.
//! - **Subscript $n$:** $\mathbf C_{p,n}^{-1}$ is the committed value at the end of the last converged
//!   step $t_n$. The return mapping of step $n+1$ starts from it and returns
//!   $\mathbf C_{p,n+1}^{-1} = \mathbf F_{n+1}^{-1}\exp(2\boldsymbol\varepsilon_e)\mathbf F_{n+1}^{-T}$, i.e. the
//!   updated $\mathbf b^e_{n+1} = \exp(2\boldsymbol\varepsilon_e)$ pulled back. Elastic steps leave it
//!   unchanged.
//! - **Initial value:** $\mathbf C_p^{-1} = \mathbf I$ in the initial, plastically undeformed
//!   state, not zero (see `initial_history`). In general, every configuration
//!   $\mathbf F = \mathbf R\thinspace\mathbf C_p^{1/2}$ with a rotation $\mathbf R$ is stress-free.
//! - **Plastic volume change:** $\det\mathbf C_p^{-1} = J_p^{-2}$ with $J_p = \det\mathbf F^p$. It stays
//!   1 for isochoric flow such as von Mises plasticity.
//! - **Storage:** the first 6 entries of the history are $\mathbf C_p^{-1}$ in Mandel notation,
//!   `[c11, c22, c33, √2 c12, √2 c13, √2 c23]` with $c_{ij}$ the components of $\mathbf C_p^{-1}$,
//!   followed by the hardening variable, see [`HenckyPlasticityHistory`].
use std::marker::PhantomData;

use nalgebra::{Matrix3, SMatrix, SVector};

use crate::create_history_parameter_struct;
use crate::finite_strain::tensor::{
    expm_sym_fast, half_log_divided_differences, half_log_pade, half_log_spectral, mandel_rotation,
    sym, sym_eigen,
};
use crate::interfaces::{ArrayEquivalent, FiniteStrainConstitutiveModelFn, QDim, StaticMap};
use crate::linear_elasticity::{LinearElasticityHistory, LinearElasticityParameters};
use crate::mandel::{isotropic_elastic_tangent, mandel_to_sym_matrix, sym_matrix_to_mandel};
use crate::finite_strain::cauchy::CauchyScaled;
use crate::plasticity::{NewtonSolverStrain, Plasticity, PlasticityState, stress_based_jacobian};

/// Kinematic quantities of one Hencky update.
pub struct HenckyKinematics {
    /// $J = \det\mathbf F$
    pub det_f: f64,
    /// deformation gradient $\mathbf F$
    pub f: Matrix3<f64>,
    /// trial elastic left Cauchy-Green tensor $\mathbf b^e_{tr} = \mathbf F\mathbf C_{p,n}^{-1}\mathbf F^T$
    pub b_tr: Matrix3<f64>,
    /// trial logarithmic elastic strain $\tfrac12\ln\mathbf b^e_{tr}$
    pub eps_tr: Matrix3<f64>,
    /// eigenvalues of $\mathbf b^e_{tr} - \mathbf I$ and eigenvectors of $\mathbf b^e_{tr}$, if they were
    /// computed
    spectral: Option<(SVector<f64, 3>, Matrix3<f64>)>,
}

impl HenckyKinematics {
    /// `f` is the row-major deformation gradient, `cp_inv` is the committed plastic history
    /// $\mathbf C_{p,n}^{-1}$ (the identity for elasticity and for a plastically undeformed state, see
    /// [Plastic history](self#plastic-history)). Without
    /// tangent, $\tfrac12\ln\mathbf b^e_{tr}$ is computed with a Padé approximant close to the
    /// identity, which avoids the eigen decomposition.
    #[inline]
    pub fn new(f: &[f64; 9], cp_inv: &Matrix3<f64>, with_tangent: bool) -> Result<Self, String> {
        let f = Matrix3::from_row_slice(f);
        let det = f.determinant();
        if !(det > 0.0) {
            return Err(format!("det F = {} is not positive", det));
        }
        let b_tr = sym(&(f * cp_inv * f.transpose()));
        if !with_tangent {
            if let Some(eps_tr) = half_log_pade(&b_tr) {
                return Ok(Self {
                    det_f: det,
                    f,
                    b_tr,
                    eps_tr,
                    spectral: None,
                });
            }
        }
        let (eps_tr, x, q) = half_log_spectral(&b_tr).ok_or_else(|| {
            "trial b^e is not positive definite; was cp_inv initialized to the identity?"
                .to_string()
        })?;
        Ok(Self {
            det_f: det,
            f,
            b_tr,
            eps_tr,
            spectral: Some((x, q)),
        })
    }

    /// $\mathbf F^{-1}$, computed on demand: only the update of $\mathbf C_p^{-1}$ after a plastic
    /// step needs it.
    #[inline]
    pub fn f_inv(&self) -> Result<Matrix3<f64>, String> {
        self.f
            .try_inverse()
            .ok_or_else(|| "F is not invertible".to_string())
    }

    /// Eigenvalues of $\mathbf b^e_{tr} - \mathbf I$ and eigenvectors of $\mathbf b^e_{tr}$, computed
    /// here if the Padé approximant was used for the trial strain.
    #[inline]
    fn spectral(&self) -> (SVector<f64, 3>, Matrix3<f64>) {
        match self.spectral {
            Some(spectral) => spectral,
            None => sym_eigen(&(self.b_tr - Matrix3::identity())),
        }
    }

    /// $\mathbb a = \partial\boldsymbol\tau/\partial\mathbf d$ in Mandel notation (entry $(i,j)$
    /// is $\partial\tau_i/\partial d_j$) from $\mathbf D = \partial\boldsymbol\tau/
    /// \partial\boldsymbol\varepsilon_{tr}$ (not transposed). With $\mathbf C_{p,n}^{-1}$ fixed,
    /// $\delta\mathbf b^e_{tr} = \mathbf l\mathbf b^e_{tr} + \mathbf b^e_{tr}\mathbf l^T$, whose
    /// symmetric part $\mathbf d\mathbf b + \mathbf b\mathbf d$ is mapped through the derivative of
    /// $\tfrac12\ln$. The skew part of $\mathbf l$ only rotates $\boldsymbol\tau$.
    ///
    /// For an isotropic elastic $\mathbf D = \mathbb C$, use the cheaper
    /// [`HenckyKinematics::elastic_kirchhoff_tangent`].
    #[inline]
    pub fn kirchhoff_tangent(&self, d: &SMatrix<f64, 6, 6>) -> SMatrix<f64, 6, 6> {
        let (x, q) = self.spectral();
        // In the eigenbasis of b, Q^T (d b + b d) Q has the entries (λ_i + λ_j) d_ij, so
        // ∂ε_tr/∂d is diagonal there: 1 for the normal components and Γ_ij (λ_i + λ_j) for the
        // shear components. R6 rotates it back.
        let [e01, e02, e12] = log_derivative_shear_factors(&x);
        let scaling = SVector::<f64, 6>::new(1.0, 1.0, 1.0, e01, e02, e12);
        let r6 = mandel_rotation(&q);
        let deps_dd = r6 * SMatrix::<f64, 6, 6>::from_diagonal(&scaling) * r6.transpose();
        d * deps_dd
    }

    /// [`HenckyKinematics::kirchhoff_tangent`] for an isotropic elastic tangent
    /// $\mathbb C = 2\mu\thinspace\mathbb P_{dev} + 3\kappa\thinspace\mathbb P_{vol}$, i.e. for elasticity and for
    /// elastic steps of plasticity. $\mathbb C$ acts on every shear component in the eigenbasis of
    /// $\mathbf b^e_{tr}$ as $2\mu$, its shear diagonal entry, so the product with
    /// $\partial\boldsymbol\varepsilon_{tr}/\partial\mathbf d$ reduces to the rank-3 update
    /// $\mathbb a = \mathbb C + 2\mu\sum_s (e_s - 1)\thinspace\mathbf r_s\otimes\mathbf r_s$ with the shear factors
    /// $e_s$ and the shear columns $\mathbf r_s$ of [`mandel_rotation`].
    #[inline]
    pub fn elastic_kirchhoff_tangent(&self, elastic_tangent: &SMatrix<f64, 6, 6>) -> SMatrix<f64, 6, 6> {
        let two_mu = elastic_tangent[(3, 3)];
        debug_assert!(
            {
                let kappa = (elastic_tangent[(0, 0)] + 2.0 * elastic_tangent[(0, 1)]) / 3.0;
                let isotropic = isotropic_elastic_tangent::<6>(0.5 * two_mu, kappa);
                (isotropic - elastic_tangent).norm() <= 1e-12 * elastic_tangent.norm()
            },
            "elastic_kirchhoff_tangent requires an isotropic elastic tangent"
        );
        let (x, q) = self.spectral();
        let r6 = mandel_rotation(&q);
        let mut a = *elastic_tangent;
        for (s, e) in log_derivative_shear_factors(&x).into_iter().enumerate() {
            let r = r6.column(3 + s);
            a += (two_mu * (e - 1.0)) * r * r.transpose();
        }
        a
    }
}

/// Shear factors $e_{ij} = \Gamma_{ij}(\lambda_i + \lambda_j)$, $(ij) = (01), (02), (12)$, of
/// $\partial\boldsymbol\varepsilon/\partial\mathbf d$ in the eigenbasis of $\mathbf b$, where
/// $\boldsymbol\varepsilon = \tfrac12\ln\mathbf b$ and $\lambda_i = 1 + x_i$ are the eigenvalues of
/// $\mathbf b$. The normal components have the factor 1.
#[inline]
fn log_derivative_shear_factors(x: &SVector<f64, 3>) -> [f64; 3] {
    let gamma = half_log_divided_differences(x);
    let lambda = x.add_scalar(1.0);
    [
        gamma[(0, 1)] * (lambda[0] + lambda[1]),
        gamma[(0, 2)] * (lambda[0] + lambda[2]),
        gamma[(1, 2)] * (lambda[1] + lambda[2]),
    ]
}

#[inline]
fn write_row_major(matrix: &SMatrix<f64, 6, 6>, output: &mut [[f64; 6]; 6]) {
    // nalgebra stores columns, so the columns of the transpose are the rows of `matrix`
    *output = matrix.transpose().data.0;
}

/// Hencky hyperelasticity $\boldsymbol\tau = \mathbb C\thinspace \tfrac12\ln\mathbf b$,
/// $\mathbf b = \mathbf F\mathbf F^T$. Parameters `mu`, `kappa` as in
/// [`crate::linear_elasticity::LinearElasticity3D`].
pub struct HenckyElasticity3D;

impl FiniteStrainConstitutiveModelFn<0, 0, 2, 2> for HenckyElasticity3D {
    type History = LinearElasticityHistory;
    type Parameters = LinearElasticityParameters;

    fn evaluate(
        _time: f64,
        _del_time: f64,
        f: &[f64; 9],
        tau: &mut [f64; 6],
        tangent: Option<&mut [[f64; 6]; 6]>,
        _history: &mut [f64; 0],
        parameters: &[f64; 2],
    ) -> Result<(), String> {
        let parameters = LinearElasticityParameters::from_array(parameters);
        let elastic_tangent = isotropic_elastic_tangent::<6>(parameters.mu, parameters.kappa);
        let kinematics = HenckyKinematics::new(f, &Matrix3::identity(), tangent.is_some())?;
        *tau = (elastic_tangent * sym_matrix_to_mandel(&kinematics.eps_tr)).data.0[0];
        if let Some(tangent) = tangent {
            write_row_major(&kinematics.elastic_kirchhoff_tangent(&elastic_tangent), tangent);
        }
        Ok(())
    }

    fn initial_history() -> [f64; 0] {
        []
    }
}

create_history_parameter_struct!(
    /// History of [`HenckyPlasticity3D`], 7 values per quadrature point:
    ///
    /// - `cp_inv`: the inverse plastic right Cauchy–Green tensor
    ///   $\mathbf C_p^{-1} = (\mathbf F^{pT}\mathbf F^p)^{-1}$ in Mandel notation, initially the identity
    ///   `[1, 1, 1, 0, 0, 0]`. It lives in the reference configuration and must never be rotated,
    ///   see [Plastic history](self#plastic-history).
    /// - `alpha`: the hardening variable of the plasticity model, e.g. the equivalent plastic
    ///   strain for von Mises plasticity, initially 0.
    HenckyPlasticityHistory,
    2,
    7,
    [(cp_inv, (QDim::Vector(6))), (alpha, (QDim::Scalar))]
);

/// Large-strain version of [`crate::plasticity::IsotropicPlasticityModel3D`] for any isotropic
/// [`Plasticity`] model. The model must not depend on anything but the current stress and
/// hardening variable.
///
/// The elastic law is always $\boldsymbol\tau = \mathbb C:\boldsymbol\varepsilon_e$. The yield
/// function, flow direction and hardening function of `MODEL` are evaluated at the Kirchhoff
/// stress $\boldsymbol\tau$ if `CAUCHY_YIELD` is `false` (the default), and at the Cauchy stress
/// $\boldsymbol\sigma = \boldsymbol\tau/J$ otherwise, see [`CauchyScaled`].
pub struct HenckyPlasticity3D<
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    MODEL: Plasticity<6, N_PARAMETERS, PARAMETERS, 1>,
    const CAUCHY_YIELD: bool = false,
>(PhantomData<MODEL>);

/// Return mapping and history update of [`HenckyPlasticity3D`]. `model` is either the plasticity
/// model itself or the model wrapped in [`CauchyScaled`]; in the latter case
/// `depends_on_volume_ratio` adds the derivative of $\boldsymbol\tau$ with respect to $J$ to
/// the tangent.
fn hencky_plastic_update<
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    MODEL: Plasticity<6, N_PARAMETERS, PARAMETERS, 1>,
>(
    model: &MODEL,
    depends_on_volume_ratio: bool,
    kinematics: &HenckyKinematics,
    history: &mut HenckyPlasticityHistory,
    tau: &mut [f64; 6],
    tangent: Option<&mut [[f64; 6]; 6]>,
) -> Result<(), String> {
    let tau_tr = model.elastic_tangent() * sym_matrix_to_mandel(&kinematics.eps_tr);
    let alpha_0 = SVector::<f64, 1>::from_element(history.alpha);

    if model.calculate_f_only(&tau_tr, &alpha_0) <= 0.0 {
        // elastic: C_p^{-1} and alpha do not change, tau does not depend on J explicitly
        *tau = tau_tr.data.0[0];
        if let Some(tangent) = tangent {
            write_row_major(&kinematics.elastic_kirchhoff_tangent(model.elastic_tangent()), tangent);
        }
        return Ok(());
    }

    let solver = NewtonSolverStrain::<8, 6, N_PARAMETERS, PARAMETERS, 1, MODEL>::new(
        model, 1e-12, 1e-12, 50,
    );
    let mut state = PlasticityState::zeros();
    let result = solver.solve(&tau_tr, &alpha_0, &mut state).map_err(|err| {
        format!(
            "return mapping failed for tau_tr {}, alpha_0 {}: {:?}",
            tau_tr.transpose(),
            alpha_0,
            err
        )
    })?;
    if let Some(tangent) = tangent {
        // inverse of the Jacobian of the residual (r_tau, r_f, r_alpha) with respect to
        // (tau, del_lambda, alpha) at the converged state
        let mut dres = SMatrix::<f64, 8, 8>::zeros();
        stress_based_jacobian(model.elastic_tangent(), &state, &mut dres, result.del_lambda);
        let inverse = dres
            .try_inverse()
            .ok_or_else(|| "algorithmic tangent is singular".to_string())?;
        let dtau_deps_tr: SMatrix<f64, 6, 6> =
            inverse.fixed_view::<6, 6>(0, 0) * model.elastic_tangent();
        let mut a = kinematics.kirchhoff_tangent(&dtau_deps_tr);
        if depends_on_volume_ratio {
            // The residual depends on J through sigma = tau/J. With the derivatives of the
            // wrapped model (already divided by J), -J dr/dJ = (del_lambda C dg/dtau tau,
            // df/dtau tau, -del_lambda dk/dtau tau) and J dtau/dJ = [dres^-1 (-J dr/dJ)]_tau.
            // Since dJ = J tr(d), this adds J dtau/dJ ⊗ I to the tangent.
            let mut rhs = SVector::<f64, 8>::zeros();
            rhs.fixed_rows_mut::<6>(0).copy_from(
                &(result.del_lambda * model.elastic_tangent() * state.dg_dsigma * result.sigma),
            );
            rhs[6] = (state.df_dsigma * result.sigma)[0];
            rhs[7] = -result.del_lambda * (state.dk_dsigma * result.sigma)[0];
            let j_dtau_dj: SVector<f64, 6> = (inverse * rhs).fixed_rows::<6>(0).into();
            let identity = SVector::<f64, 6>::new(1.0, 1.0, 1.0, 0.0, 0.0, 0.0);
            a += j_dtau_dj * identity.transpose();
        }
        write_row_major(&a, tangent);
    }
    // C_p^{-1} = F^{-1} exp(2 eps_e) F^{-T}
    let eps_e = kinematics.eps_tr - mandel_to_sym_matrix(&result.del_plastic_strain);
    let b_e = expm_sym_fast(&(2.0 * eps_e));
    let f_inv = kinematics.f_inv()?;
    let cp_inv = sym(&(f_inv * b_e * f_inv.transpose()));
    history.cp_inv = sym_matrix_to_mandel(&cp_inv);
    history.alpha = result.kappa[0];
    *tau = result.sigma.data.0[0];
    Ok(())
}

impl<
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    MODEL: Plasticity<6, N_PARAMETERS, PARAMETERS, 1>,
    const CAUCHY_YIELD: bool,
> FiniteStrainConstitutiveModelFn<2, 7, N_PARAMETERS, PARAMETERS>
    for HenckyPlasticity3D<N_PARAMETERS, PARAMETERS, MODEL, CAUCHY_YIELD>
{
    type History = HenckyPlasticityHistory;
    type Parameters = MODEL::Parameters;

    fn evaluate(
        _time: f64,
        _del_time: f64,
        f: &[f64; 9],
        tau: &mut [f64; 6],
        tangent: Option<&mut [[f64; 6]; 6]>,
        history: &mut [f64; 7],
        parameters: &[f64; PARAMETERS],
    ) -> Result<(), String> {
        let history = HenckyPlasticityHistory::from_array_mut(history);
        let model = MODEL::new(Self::Parameters::from_array(parameters));
        let kinematics =
            HenckyKinematics::new(f, &mandel_to_sym_matrix(&history.cp_inv), tangent.is_some())?;
        if CAUCHY_YIELD {
            let model = CauchyScaled::with_volume_ratio(model, kinematics.det_f);
            hencky_plastic_update(&model, true, &kinematics, history, tau, tangent)
        } else {
            hencky_plastic_update(&model, false, &kinematics, history, tau, tangent)
        }
    }

    fn initial_history() -> [f64; 7] {
        let history = HenckyPlasticityHistory {
            cp_inv: SVector::<f64, 6>::new(1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
            alpha: 0.0,
        };
        *history.as_array()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interfaces::ConstitutiveModelFn;
    use crate::linear_elasticity::LinearElasticity3D;
    use crate::plasticity::{
        DruckerPrager3D, DruckerPragerHyperbolic3D, DruckerPragerParameters, IsotropicMises3D,
        IsotropicMisesParameters, IsotropicPlasticityModel3D,
    };
    use nalgebra::{Rotation3, Unit, Vector3};

    type HenckyMises = HenckyPlasticity3D<4, 4, IsotropicMises3D>;
    type HenckyDP = HenckyPlasticity3D<5, 5, DruckerPrager3D>;
    type HenckyDPH = HenckyPlasticity3D<7, 7, DruckerPragerHyperbolic3D>;
    type CauchyMises = HenckyPlasticity3D<4, 4, IsotropicMises3D, true>;
    type CauchyDP = HenckyPlasticity3D<5, 5, DruckerPrager3D, true>;
    type CauchyDPH = HenckyPlasticity3D<7, 7, DruckerPragerHyperbolic3D, true>;
    type SmallMises = IsotropicPlasticityModel3D<4, 4, IsotropicMises3D>;
    type SmallDP = IsotropicPlasticityModel3D<5, 5, DruckerPrager3D>;
    type SmallDPH = IsotropicPlasticityModel3D<7, 7, DruckerPragerHyperbolic3D>;

    const MU: f64 = 80.77;
    const KAPPA: f64 = 166.67;
    const ELASTIC: [f64; 2] = [MU, KAPPA];
    const MISES: [f64; 4] = [MU, KAPPA, 0.2, 10.0];
    const DP: [f64; 5] = [MU, KAPPA, 0.15, 0.05, 0.03];
    const DPH: [f64; 7] = [MU, KAPPA, 0.15, 0.05, 0.1, 5.0, 0.03];

    struct Output {
        tau: Matrix3<f64>,
        tangent: SMatrix<f64, 6, 6>,
    }

    fn row_major(matrix: &Matrix3<f64>) -> [f64; 9] {
        let mut output = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                output[3 * i + j] = matrix[(i, j)];
            }
        }
        output
    }

    fn evaluate<
        const NH: usize,
        const H: usize,
        const NP: usize,
        const P: usize,
        M: FiniteStrainConstitutiveModelFn<NH, H, NP, P>,
    >(
        f: &Matrix3<f64>,
        history: &mut [f64; H],
        parameters: &[f64; P],
    ) -> Result<Output, String> {
        let mut tau = [0.0; 6];
        let mut tangent = [[0.0; 6]; 6];
        M::evaluate(0.0, 0.0, &row_major(f), &mut tau, Some(&mut tangent), history, parameters)?;
        Ok(Output {
            tau: mandel_to_sym_matrix(&SVector::<f64, 6>::from_column_slice(&tau)),
            tangent: SMatrix::<f64, 6, 6>::from_row_iterator(tangent.iter().flatten().copied()),
        })
    }

    /// Checks the tangent with central differences along all nine directions
    /// $\delta\mathbf F = \mathbf l\mathbf F$: the six symmetric $\mathbf l = \mathbf d$ give the
    /// columns of $\mathbb a$, the three skew $\mathbf l = \mathbf w$ must give
    /// $\mathbf w\boldsymbol\tau - \boldsymbol\tau\mathbf w$. Returns true if the step was plastic.
    fn check_tangent<
        const NH: usize,
        const H: usize,
        const NP: usize,
        const P: usize,
        M: FiniteStrainConstitutiveModelFn<NH, H, NP, P>,
    >(
        f: &Matrix3<f64>,
        history_0: &[f64; H],
        parameters: &[f64; P],
    ) -> bool {
        // 1e-7 is already dominated by round-off in tau near the identity
        let step = 1e-6;
        let mut history = *history_0;
        let out = evaluate::<NH, H, NP, P, M>(f, &mut history, parameters).unwrap();
        let plastic = history != *history_0;
        let derivative = |l: &Matrix3<f64>| {
            let plus = evaluate::<NH, H, NP, P, M>(&(f + step * l * f), &mut history_0.clone(), parameters)
                .unwrap()
                .tau;
            let minus = evaluate::<NH, H, NP, P, M>(&(f - step * l * f), &mut history_0.clone(), parameters)
                .unwrap()
                .tau;
            (plus - minus) / (2.0 * step)
        };

        let mut fd = SMatrix::<f64, 6, 6>::zeros();
        for k in 0..6 {
            let d = mandel_to_sym_matrix(&SVector::<f64, 6>::from_fn(|i, _| {
                if i == k { 1.0 } else { 0.0 }
            }));
            fd.set_column(k, &sym_matrix_to_mandel(&derivative(&d)));
        }
        let error = (out.tangent - fd).abs().max() / out.tangent.abs().max();
        assert!(
            error < 1e-6,
            "relative tangent error {:e}\nanalytical {}\nfd {}",
            error,
            out.tangent,
            fd
        );

        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            let mut w = Matrix3::<f64>::zeros();
            w[(i, j)] = 1.0;
            w[(j, i)] = -1.0;
            let expected = w * out.tau - out.tau * w;
            let scale = out.tau.abs().max().max(1e-6 * out.tangent.abs().max());
            let error = (derivative(&w) - expected).abs().max() / scale;
            assert!(error < 1e-6, "spin derivative error {:e}", error);
        }
        plastic
    }

    fn rotation(axis: Vector3<f64>, angle: f64) -> Matrix3<f64> {
        *Rotation3::from_axis_angle(&Unit::new_normalize(axis), angle).matrix()
    }

    /// general deformation gradients with stretch, shear and rotation
    fn f_general_1() -> Matrix3<f64> {
        Matrix3::new(1.02, 0.01, -0.005, 0.004, 0.99, 0.008, -0.003, 0.006, 0.995)
    }
    fn f_general_2() -> Matrix3<f64> {
        rotation(Vector3::new(0.2, 1.0, -0.4), 0.3)
            * Matrix3::new(1.05, 0.03, 0.0, 0.01, 0.97, 0.02, 0.0, -0.01, 0.99)
    }
    fn f_compression_1() -> Matrix3<f64> {
        Matrix3::new(0.98, 0.004, 0.0, 0.002, 1.004, 0.001, 0.0, 0.003, 1.003)
    }
    fn f_compression_2() -> Matrix3<f64> {
        rotation(Vector3::new(1.0, 0.3, 0.2), 0.4)
            * Matrix3::new(0.95, 0.02, 0.01, 0.0, 1.01, 0.0, 0.01, 0.0, 1.012)
    }

    fn mises_history_after(f: &Matrix3<f64>) -> [f64; 7] {
        let mut history = HenckyMises::initial_history();
        evaluate::<2, 7, 4, 4, HenckyMises>(f, &mut history, &MISES).unwrap();
        history
    }

    #[test]
    fn test_elasticity_analytic() {
        let lambda = Vector3::new(1.5, 0.8, 1.1);
        let f = Matrix3::from_diagonal(&lambda);
        let tau = evaluate::<0, 0, 2, 2, HenckyElasticity3D>(&f, &mut [], &ELASTIC)
            .unwrap()
            .tau;
        let log_v = lambda.map(f64::ln);
        let ln_j = log_v.sum();
        let expected = Matrix3::from_diagonal(&log_v.map(|e| 2.0 * MU * (e - ln_j / 3.0)))
            + KAPPA * ln_j * Matrix3::identity();
        assert!((tau - expected).norm() <= 1e-12 * expected.norm());
    }

    #[test]
    fn test_tangent_elasticity() {
        for f in [f_general_1(), f_general_2(), Matrix3::identity()] {
            check_tangent::<0, 0, 2, 2, HenckyElasticity3D>(&f, &[], &ELASTIC);
        }
        // at the undeformed state the tangent is the elastic tangent
        let out = evaluate::<0, 0, 2, 2, HenckyElasticity3D>(&Matrix3::identity(), &mut [], &ELASTIC)
            .unwrap();
        let elastic_tangent = isotropic_elastic_tangent::<6>(MU, KAPPA);
        assert!((out.tangent - elastic_tangent).norm() <= 1e-14 * elastic_tangent.norm());
    }

    #[test]
    fn test_tangent_mises() {
        let initial = HenckyMises::initial_history();
        assert!(!check_tangent::<2, 7, 4, 4, HenckyMises>(
            &(Matrix3::identity() + 1e-4 * f_general_1()),
            &initial,
            &MISES
        ));
        assert!(check_tangent::<2, 7, 4, 4, HenckyMises>(&f_general_1(), &initial, &MISES));
        // non-identity C_p^{-1} from a preceding plastic step
        let history = mises_history_after(&f_general_1());
        assert!(check_tangent::<2, 7, 4, 4, HenckyMises>(&f_general_2(), &history, &MISES));
        // large stretch in one step
        let f = Matrix3::from_diagonal(&Vector3::new(1.8, 0.8, 0.75));
        assert!(check_tangent::<2, 7, 4, 4, HenckyMises>(&f, &history, &MISES));
    }

    #[test]
    fn test_tangent_drucker_prager() {
        let initial = HenckyDP::initial_history();
        assert!(!check_tangent::<2, 7, 5, 5, HenckyDP>(
            &Matrix3::from_diagonal(&Vector3::new(0.9998, 1.0, 1.0)),
            &initial,
            &DP
        ));
        assert!(check_tangent::<2, 7, 5, 5, HenckyDP>(&f_compression_1(), &initial, &DP));
        let mut history = initial;
        evaluate::<2, 7, 5, 5, HenckyDP>(&f_compression_1(), &mut history, &DP).unwrap();
        assert!(check_tangent::<2, 7, 5, 5, HenckyDP>(&f_compression_2(), &history, &DP));
    }

    #[test]
    fn test_tangent_drucker_prager_hyperbolic() {
        let initial = HenckyDPH::initial_history();
        assert!(!check_tangent::<2, 7, 7, 7, HenckyDPH>(
            &Matrix3::from_diagonal(&Vector3::new(0.9998, 1.0, 1.0)),
            &initial,
            &DPH
        ));
        assert!(check_tangent::<2, 7, 7, 7, HenckyDPH>(&f_compression_1(), &initial, &DPH));
        let mut history = initial;
        evaluate::<2, 7, 7, 7, HenckyDPH>(&f_compression_1(), &mut history, &DPH).unwrap();
        assert!(check_tangent::<2, 7, 7, 7, HenckyDPH>(&f_compression_2(), &history, &DPH));
        assert!(check_tangent::<2, 7, 7, 7, HenckyDPH>(&f_general_2(), &history, &DPH));
    }

    /// Drives the large-strain model along a coaxial path F = diag(λ(t)) and the small-strain
    /// model with Δε = ln(λ_{n+1}/λ_n). Both algorithms are identical for coaxial paths, so the
    /// Kirchhoff stress and the hardening variable must agree to round-off.
    fn check_coaxial<
        const NP: usize,
        const P: usize,
        Large: FiniteStrainConstitutiveModelFn<2, 7, NP, P>,
        Small: ConstitutiveModelFn<6, 2, 7, NP, P>,
    >(
        path: &[Vector3<f64>],
        parameters: &[f64; P],
    ) {
        let mut history_large = Large::initial_history();
        let mut history_small = [0.0; 7];
        let mut stress_small = [0.0; 6];
        let mut lambda_n = Vector3::new(1.0, 1.0, 1.0);
        let mut max_alpha: f64 = 0.0;
        for lambda in path {
            let f = Matrix3::from_diagonal(lambda);
            let tau = evaluate::<2, 7, NP, P, Large>(&f, &mut history_large, parameters)
                .unwrap()
                .tau;
            let del_eps = lambda.component_div(&lambda_n).map(f64::ln);
            let mut tangent = [[0.0; 6]; 6];
            Small::evaluate(
                0.0,
                0.0,
                &[del_eps[0], del_eps[1], del_eps[2], 0.0, 0.0, 0.0],
                &mut stress_small,
                Some(&mut tangent),
                &mut history_small,
                parameters,
            );
            lambda_n = *lambda;
            let sigma = mandel_to_sym_matrix(&SVector::<f64, 6>::from_column_slice(&stress_small));
            assert!(
                (tau - sigma).norm() <= 1e-10 * sigma.norm(),
                "lambda {}: tau {} sigma {}",
                lambda.transpose(),
                tau,
                sigma
            );
            let (alpha_large, alpha_small) = (history_large[6], history_small[0]);
            assert!((alpha_large - alpha_small).abs() <= 1e-10 * alpha_small.abs().max(1e-12));
            max_alpha = max_alpha.max(alpha_small);
        }
        assert!(max_alpha > 0.0, "path did not reach plasticity");
    }

    /// λ_1 from 1 to `max`, back to `back`, with lateral stretches λ_1^(-0.3) and λ_1^(-0.4)
    fn coaxial_path(max: f64, back: f64, n_steps: usize) -> Vec<Vector3<f64>> {
        let lateral = |l: f64| Vector3::new(l, l.powf(-0.3), l.powf(-0.4));
        let mut path: Vec<Vector3<f64>> = (1..=n_steps)
            .map(|i| lateral(1.0 + (max - 1.0) * i as f64 / n_steps as f64))
            .collect();
        path.extend((1..=n_steps / 2).map(|i| {
            lateral(max + (back - max) * i as f64 / (n_steps / 2) as f64)
        }));
        path
    }

    #[test]
    fn test_coaxial_equivalence_mises() {
        check_coaxial::<4, 4, HenckyMises, SmallMises>(&coaxial_path(2.0, 1.5, 20), &MISES);
        check_coaxial::<4, 4, HenckyMises, SmallMises>(&coaxial_path(0.5, 0.7, 20), &MISES);
    }

    #[test]
    fn test_coaxial_equivalence_drucker_prager() {
        check_coaxial::<5, 5, HenckyDP, SmallDP>(&coaxial_path(0.5, 0.6, 20), &DP);
        check_coaxial::<7, 7, HenckyDPH, SmallDPH>(&coaxial_path(0.5, 0.6, 20), &DPH);
        check_coaxial::<7, 7, HenckyDPH, SmallDPH>(&coaxial_path(2.0, 1.5, 20), &DPH);
    }

    #[test]
    fn test_objectivity() {
        let q = rotation(Vector3::new(-0.5, 0.2, 1.0), 1.1);
        let history_0 = mises_history_after(&f_general_1());
        let f = f_general_2();
        let mut history = history_0;
        let out = evaluate::<2, 7, 4, 4, HenckyMises>(&f, &mut history, &MISES).unwrap();
        let mut history_rotated = history_0;
        let out_rotated =
            evaluate::<2, 7, 4, 4, HenckyMises>(&(q * f), &mut history_rotated, &MISES).unwrap();
        assert!(history[6] > history_0[6], "step is not plastic");
        assert!((out_rotated.tau - q * out.tau * q.transpose()).norm() <= 1e-12 * out.tau.norm());
        for i in 0..7 {
            assert!((history_rotated[i] - history[i]).abs() <= 1e-12);
        }
    }

    #[test]
    fn test_small_strain_limit() {
        let eps = 1e-6;
        let h = Matrix3::new(0.3, -0.2, 0.5, 0.7, 1.0, 0.1, -0.4, 0.2, -0.4);
        let f = Matrix3::identity() + eps * h;
        let mut stress = [0.0; 6];
        let del_strain = sym_matrix_to_mandel(&sym(&(eps * h)));
        LinearElasticity3D::evaluate(
            0.0,
            0.0,
            &del_strain.data.0[0],
            &mut stress,
            None,
            &mut [],
            &ELASTIC,
        );
        let sigma = mandel_to_sym_matrix(&SVector::<f64, 6>::from_column_slice(&stress));
        let tau_elastic = evaluate::<0, 0, 2, 2, HenckyElasticity3D>(&f, &mut [], &ELASTIC)
            .unwrap()
            .tau;
        let mut history = HenckyMises::initial_history();
        let tau_mises = evaluate::<2, 7, 4, 4, HenckyMises>(&f, &mut history, &MISES)
            .unwrap()
            .tau;
        assert!((tau_elastic - sigma).norm() <= 10.0 * eps * sigma.norm());
        assert!((tau_mises - sigma).norm() <= 10.0 * eps * sigma.norm());
    }

    #[test]
    fn test_errors() {
        let mut history = HenckyMises::initial_history();
        let inverted = Matrix3::from_diagonal(&Vector3::new(-1.0, 1.0, 1.0));
        assert!(evaluate::<2, 7, 4, 4, HenckyMises>(&inverted, &mut history, &MISES).is_err());
        assert!(evaluate::<0, 0, 2, 2, HenckyElasticity3D>(&inverted, &mut [], &ELASTIC).is_err());
        let mut zero_history = [0.0; 7];
        assert!(
            evaluate::<2, 7, 4, 4, HenckyMises>(&f_general_1(), &mut zero_history, &MISES)
                .is_err()
        );
    }

    #[test]
    fn test_evaluate_all_points() {
        let f = [row_major(&f_general_1()), row_major(&f_general_2())].concat();
        let mut tau = vec![0.0; 12];
        let mut tangent = vec![0.0; 72];
        let mut history = [HenckyMises::initial_history(), HenckyMises::initial_history()].concat();
        crate::interfaces::evaluate_finite_strain_model::<2, 7, 4, 4, HenckyMises>(
            0.0,
            0.0,
            &f,
            &mut tau,
            Some(&mut tangent),
            &mut history,
            &MISES,
        )
        .unwrap();
        let mut history_1 = HenckyMises::initial_history();
        let single = evaluate::<2, 7, 4, 4, HenckyMises>(&f_general_2(), &mut history_1, &MISES)
            .unwrap();
        let tau_1 = mandel_to_sym_matrix(&SVector::<f64, 6>::from_column_slice(&tau[6..]));
        assert!((tau_1 - single.tau).norm() == 0.0);
        assert_eq!(&history[7..], &history_1);
        // wrong lengths are an error, not a panic
        assert!(
            crate::interfaces::evaluate_finite_strain_model::<2, 7, 4, 4, HenckyMises>(
                0.0,
                0.0,
                &f,
                &mut tau,
                None,
                &mut history[..7],
                &MISES,
            )
            .is_err()
        );
    }

    #[test]
    fn test_tangent_cauchy_yield() {
        let initial = CauchyMises::initial_history();
        assert!(check_tangent::<2, 7, 4, 4, CauchyMises>(&f_general_1(), &initial, &MISES));
        let mut history = initial;
        evaluate::<2, 7, 4, 4, CauchyMises>(&f_general_1(), &mut history, &MISES).unwrap();
        assert!(check_tangent::<2, 7, 4, 4, CauchyMises>(&f_general_2(), &history, &MISES));
        let f = Matrix3::from_diagonal(&Vector3::new(1.8, 0.8, 0.75));
        assert!(check_tangent::<2, 7, 4, 4, CauchyMises>(&f, &history, &MISES));

        let initial = CauchyDP::initial_history();
        assert!(check_tangent::<2, 7, 5, 5, CauchyDP>(&f_compression_1(), &initial, &DP));
        let mut history = initial;
        evaluate::<2, 7, 5, 5, CauchyDP>(&f_compression_1(), &mut history, &DP).unwrap();
        assert!(check_tangent::<2, 7, 5, 5, CauchyDP>(&f_compression_2(), &history, &DP));

        let initial = CauchyDPH::initial_history();
        assert!(check_tangent::<2, 7, 7, 7, CauchyDPH>(&f_compression_1(), &initial, &DPH));
        let mut history = initial;
        evaluate::<2, 7, 7, 7, CauchyDPH>(&f_compression_1(), &mut history, &DPH).unwrap();
        assert!(check_tangent::<2, 7, 7, 7, CauchyDPH>(&f_compression_2(), &history, &DPH));
        assert!(check_tangent::<2, 7, 7, 7, CauchyDPH>(&f_general_2(), &history, &DPH));
    }

    /// After a plastic step, the yield function vanishes at the Cauchy stress tau/J for the
    /// Cauchy variant and at tau for the Kirchhoff variant.
    #[test]
    fn test_cauchy_yield_condition() {
        let f = Matrix3::from_diagonal(&Vector3::new(1.3, 0.95, 0.9));
        let j = f.determinant();
        let mises = IsotropicMises3D::new(IsotropicMisesParameters::from_array(&MISES));
        for cauchy in [true, false] {
            let mut history = CauchyMises::initial_history();
            let tau = if cauchy {
                evaluate::<2, 7, 4, 4, CauchyMises>(&f, &mut history, &MISES)
            } else {
                evaluate::<2, 7, 4, 4, HenckyMises>(&f, &mut history, &MISES)
            }
            .unwrap()
            .tau;
            let alpha = SVector::<f64, 1>::from_element(history[6]);
            let f_cauchy = mises.calculate_f_only(&(sym_matrix_to_mandel(&tau) / j), &alpha);
            let f_kirchhoff = mises.calculate_f_only(&sym_matrix_to_mandel(&tau), &alpha);
            let (f_yield, f_other) = if cauchy { (f_cauchy, f_kirchhoff) } else { (f_kirchhoff, f_cauchy) };
            assert!(alpha[0] > 0.0);
            assert!(f_yield.abs() <= 1e-10 * MISES[2], "f = {:e}", f_yield);
            assert!(f_other.abs() > 1e-3 * MISES[2]);
        }

        let f = f_compression_2();
        let j = f.determinant();
        let dp = DruckerPrager3D::new(DruckerPragerParameters::from_array(&DP));
        let mut history = CauchyDP::initial_history();
        let tau = evaluate::<2, 7, 5, 5, CauchyDP>(&f, &mut history, &DP).unwrap().tau;
        let alpha = SVector::<f64, 1>::from_element(history[6]);
        assert!(alpha[0] > 0.0);
        assert!(dp.calculate_f_only(&(sym_matrix_to_mandel(&tau) / j), &alpha).abs() <= 1e-10 * DP[2]);
    }

    /// For det F = 1 the Cauchy and Kirchhoff stresses coincide.
    #[test]
    fn test_cauchy_yield_isochoric() {
        let mut history_cauchy = CauchyMises::initial_history();
        let mut history_kirchhoff = HenckyMises::initial_history();
        for lambda in [1.1, 1.3, 1.6, 1.4] {
            let f = rotation(Vector3::new(0.1, 0.4, 1.0), 0.2 * lambda)
                * Matrix3::from_diagonal(&Vector3::new(lambda, lambda.powf(-0.3), lambda.powf(-0.7)));
            let tau_cauchy = evaluate::<2, 7, 4, 4, CauchyMises>(&f, &mut history_cauchy, &MISES)
                .unwrap()
                .tau;
            let tau_kirchhoff =
                evaluate::<2, 7, 4, 4, HenckyMises>(&f, &mut history_kirchhoff, &MISES)
                    .unwrap()
                    .tau;
            assert!((tau_cauchy - tau_kirchhoff).norm() <= 1e-10 * tau_kirchhoff.norm());
        }
        assert!(history_cauchy[6] > 0.0);
    }

    /// Stress and history without tangent (Padé approximants close to the identity) agree with
    /// the evaluation with tangent (spectral decomposition) along whole loading paths.
    fn check_without_tangent<
        const NP: usize,
        const P: usize,
        M: FiniteStrainConstitutiveModelFn<2, 7, NP, P>,
    >(
        path: &[Matrix3<f64>],
        parameters: &[f64; P],
    ) {
        let mut history_with = M::initial_history();
        let mut history_without = M::initial_history();
        for f in path {
            let tau_with = evaluate::<2, 7, NP, P, M>(f, &mut history_with, parameters).unwrap().tau;
            let mut tau_without = [0.0; 6];
            M::evaluate(0.0, 0.0, &row_major(f), &mut tau_without, None, &mut history_without, parameters)
                .unwrap();
            let tau_without = mandel_to_sym_matrix(&SVector::<f64, 6>::from_column_slice(&tau_without));
            assert!((tau_with - tau_without).norm() <= 1e-12 * tau_with.norm().max(1e-12));
            for i in 0..7 {
                assert!((history_with[i] - history_without[i]).abs() <= 1e-12);
            }
        }
        assert!(history_with[6] > 0.0, "path did not reach plasticity");
    }

    #[test]
    fn test_without_tangent() {
        let coaxial = |max: f64, back: f64| -> Vec<Matrix3<f64>> {
            coaxial_path(max, back, 20).iter().map(Matrix3::from_diagonal).collect()
        };
        let general: Vec<Matrix3<f64>> = (1..=20)
            .map(|i| {
                let t = i as f64 / 20.0;
                rotation(Vector3::new(0.2, 1.0, -0.4), t) * (Matrix3::identity() + t * (f_general_2() - Matrix3::identity()) * 4.0)
            })
            .collect();
        check_without_tangent::<4, 4, HenckyMises>(&coaxial(2.0, 1.5), &MISES);
        check_without_tangent::<4, 4, HenckyMises>(&general, &MISES);
        check_without_tangent::<4, 4, CauchyMises>(&general, &MISES);
        check_without_tangent::<5, 5, HenckyDP>(&coaxial(0.5, 0.6), &DP);
        check_without_tangent::<7, 7, HenckyDPH>(&coaxial(0.5, 0.6), &DPH);
        check_without_tangent::<7, 7, CauchyDPH>(&coaxial(0.5, 0.6), &DPH);

        // elastic stress close to and far from the identity
        for f in [Matrix3::identity() + 1e-4 * f_general_1(), f_general_2()] {
            let tau_with = evaluate::<0, 0, 2, 2, HenckyElasticity3D>(&f, &mut [], &ELASTIC).unwrap().tau;
            let mut tau_without = [0.0; 6];
            HenckyElasticity3D::evaluate(0.0, 0.0, &row_major(&f), &mut tau_without, None, &mut [], &ELASTIC)
                .unwrap();
            let tau_without = mandel_to_sym_matrix(&SVector::<f64, 6>::from_column_slice(&tau_without));
            assert!((tau_with - tau_without).norm() <= 1e-12 * tau_with.norm());
        }
    }

    use crate::finite_strain::tensor::{expm_sym, PADE_LOG_RADIUS};
    use crate::finite_strain::test_utils::{seeded, TensorSampling};
    use rand::Rng;

    /// Random deformation gradient F = R_e V_e F_p with a large plastic part F_p, an elastic
    /// stretch with |b_e - I| = `elastic_norm`, and the matching history C_p^{-1} = (F_p^T F_p)^{-1}.
    /// The trial b_tr = F C_p^{-1} F^T = R_e V_e^2 R_e^T equals b_e up to round-off.
    fn random_state(rng: &mut impl Rng, elastic_norm: f64) -> (Matrix3<f64>, Matrix3<f64>) {
        let plastic_norm = rng.log_uniform(1e-3, 1.0);
        let s_p = rng.symmetric(plastic_norm);
        let f_p = rng.rotation() * expm_sym(&s_p);
        let x = rng.symmetric(elastic_norm);
        let (eps_e, _, _) = half_log_spectral(&(Matrix3::identity() + x)).unwrap();
        let f = rng.rotation() * expm_sym(&eps_e) * f_p;
        let cp_inv = (f_p.transpose() * f_p).try_inverse().unwrap();
        (f, sym(&cp_inv))
    }

    #[test]
    fn test_random_kinematics_pade_vs_spectral() {
        let mut rng = seeded(11);
        let mut max_error: f64 = 0.0;
        for i in 0..10_000 {
            let elastic_norm = if i % 2 == 0 {
                rng.log_uniform(1e-10, 0.98 * PADE_LOG_RADIUS)
            } else {
                PADE_LOG_RADIUS * (0.9 + 0.08 * rng.gen_range(0.0..1.0))
            };
            let (f, cp_inv) = random_state(&mut rng, elastic_norm);
            let f = row_major(&f);
            let pade = HenckyKinematics::new(&f, &cp_inv, false).unwrap();
            let spectral = HenckyKinematics::new(&f, &cp_inv, true).unwrap();
            assert!(pade.spectral.is_none(), "Pade approximant not used");
            assert!(spectral.spectral.is_some());
            let difference = (pade.eps_tr - spectral.eps_tr).norm();
            if difference > 0.0 {
                max_error = max_error.max(difference / spectral.eps_tr.norm());
            }
        }
        assert!(max_error <= 5e-15, "max. relative error {:e}", max_error);
    }

    /// Stress and history without tangent (Padé) agree with the evaluation with tangent
    /// (spectral) at random states with large plastic deformations and rotations, for elastic
    /// and plastic steps.
    fn check_random_without_tangent<
        const NP: usize,
        const P: usize,
        M: FiniteStrainConstitutiveModelFn<2, 7, NP, P>,
    >(
        seed: u64,
        parameters: &[f64; P],
    ) {
        let mut rng = seeded(seed);
        let (mut n_elastic, mut n_plastic) = (0, 0);
        for i in 0..2_000 {
            // log-uniform over many orders of magnitude (mostly elastic) and uniform in the upper
            // part of the Padé radius (mostly plastic)
            let elastic_norm = if i % 2 == 0 {
                rng.log_uniform(1e-8, 0.98 * PADE_LOG_RADIUS)
            } else {
                PADE_LOG_RADIUS * rng.gen_range(0.2..0.98)
            };
            let (f, cp_inv) = random_state(&mut rng, elastic_norm);
            let mut history = [0.0; 7];
            history[..6].copy_from_slice(sym_matrix_to_mandel(&cp_inv).as_slice());
            history[6] = 0.5 * rng.gen_range(0.0..1.0);
            let mut history_with = history;
            let mut history_without = history;
            let tau_with = evaluate::<2, 7, NP, P, M>(&f, &mut history_with, parameters)
                .unwrap()
                .tau;
            let mut tau_without = [0.0; 6];
            M::evaluate(0.0, 0.0, &row_major(&f), &mut tau_without, None, &mut history_without, parameters)
                .unwrap();
            let tau_without =
                mandel_to_sym_matrix(&SVector::<f64, 6>::from_column_slice(&tau_without));
            assert!((tau_with - tau_without).norm() <= 1e-11 * tau_with.norm().max(1e-12));
            for i in 0..7 {
                assert!((history_with[i] - history_without[i]).abs() <= 1e-11 * history_with[i].abs().max(1.0));
            }
            if history_with[6] > history[6] { n_plastic += 1 } else { n_elastic += 1 }
        }
        assert!(n_elastic > 100 && n_plastic > 100, "elastic {}, plastic {}", n_elastic, n_plastic);
    }

    #[test]
    fn test_random_without_tangent() {
        check_random_without_tangent::<4, 4, HenckyMises>(21, &MISES);
        check_random_without_tangent::<4, 4, CauchyMises>(22, &MISES);
        check_random_without_tangent::<7, 7, HenckyDPH>(23, &DPH);
        check_random_without_tangent::<7, 7, CauchyDPH>(24, &DPH);
    }

    /// Simple shear F = I + γ e_1 ⊗ e_2 has the closed-form Hencky stress
    /// τ = 2μ asinh(γ/2) [cos 2θ (e_1e_1 - e_2e_2) + sin 2θ (e_1e_2 + e_2e_1)] with
    /// cos 2θ = γ/sqrt(γ²+4), sin 2θ = 2/sqrt(γ²+4).
    #[test]
    fn test_simple_shear_analytic() {
        for gamma in [1e-3, 0.1, 1.0, 3.0, 8.0, 20.0] {
            let mut f = Matrix3::identity();
            f[(0, 1)] = gamma;
            let tau = evaluate::<0, 0, 2, 2, HenckyElasticity3D>(&f, &mut [], &ELASTIC).unwrap().tau;
            let amplitude = 2.0 * MU * (gamma / 2.0_f64).asinh() / (gamma * gamma + 4.0).sqrt();
            let expected = Matrix3::new(
                amplitude * gamma, 2.0 * amplitude, 0.0,
                2.0 * amplitude, -amplitude * gamma, 0.0,
                0.0, 0.0, 0.0,
            );
            // b = F F^T is formed with entries close to 1, so b - I carries a relative
            // rounding error of about eps/gamma
            assert!((tau - expected).norm() <= 1e-12 * expected.norm(), "gamma {}", gamma);
        }
    }

    /// Hencky elasticity is hyperelastic: the work ∫ P : dF along any path equals the strain
    /// energy ψ = μ |dev h|² + κ/2 (tr h)², h = ln V, and the stress vanishes after a closed cycle.
    #[test]
    fn test_hyperelastic_work() {
        let path_work = |corners: &[Matrix3<f64>], steps: usize| -> (f64, Matrix3<f64>) {
            let piola = |f: &Matrix3<f64>| {
                let tau = evaluate::<0, 0, 2, 2, HenckyElasticity3D>(f, &mut [], &ELASTIC).unwrap().tau;
                tau * f.try_inverse().unwrap().transpose()
            };
            let mut work = 0.0;
            for leg in corners.windows(2) {
                // Simpson's rule on every leg (P is smooth along a straight leg)
                let df = (leg[1] - leg[0]) / steps as f64;
                for k in 0..steps {
                    let f0 = leg[0] + df * k as f64;
                    let p = piola(&f0) + 4.0 * piola(&(f0 + 0.5 * df)) + piola(&(f0 + df));
                    work += (p / 6.0).component_mul(&df).sum();
                }
            }
            let last = corners.last().unwrap();
            (work, evaluate::<0, 0, 2, 2, HenckyElasticity3D>(last, &mut [], &ELASTIC).unwrap().tau)
        };
        let f_end = f_general_2() * Matrix3::from_diagonal(&Vector3::new(1.3, 0.8, 1.1));
        let (log_v, _, _) = half_log_spectral(&(f_end * f_end.transpose())).unwrap();
        let trace = log_v.trace();
        let dev = log_v - trace / 3.0 * Matrix3::identity();
        let energy = MU * dev.norm_squared() + 0.5 * KAPPA * trace * trace;

        let id = Matrix3::identity();
        let detour = id + Matrix3::new(0.3, -0.2, 0.1, 0.4, -0.1, 0.0, 0.2, 0.1, 0.5);
        for corners in [vec![id, f_end], vec![id, detour, f_end]] {
            let (work, _) = path_work(&corners, 200);
            assert!((work - energy).abs() <= 1e-9 * energy, "work {} energy {}", work, energy);
        }
        // closed cycle: stretch, shear, unstretch, unshear
        let stretched = Matrix3::from_diagonal(&Vector3::new(1.5, 1.0 / 1.5, 1.0));
        let sheared = stretched + Matrix3::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let unstretched = id + Matrix3::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let (work, tau) = path_work(&[id, stretched, sheared, unstretched, id], 200);
        assert!(work.abs() <= 1e-10 * energy, "work over a closed cycle {}", work);
        assert!(tau.norm() == 0.0);
    }

    /// Von Mises flow is deviatoric and the exponential map preserves the plastic volume
    /// exactly: det C_p^{-1} = 1 along random non-coaxial paths with large deformations.
    #[test]
    fn test_plastic_incompressibility() {
        for (seed, cauchy) in [(31, false), (32, true)] {
            let mut rng = seeded(seed);
            let mut history = HenckyMises::initial_history();
            let mut f = Matrix3::identity();
            for _ in 0..100 {
                let stretch = rng.symmetric(0.03);
                // stretch increment in a random direction followed by a random rigid rotation
                f = rng.rotation() * expm_sym(&stretch) * f;
                let tau = if cauchy {
                    evaluate::<2, 7, 4, 4, CauchyMises>(&f, &mut history, &MISES)
                } else {
                    evaluate::<2, 7, 4, 4, HenckyMises>(&f, &mut history, &MISES)
                }
                .unwrap()
                .tau;
                assert!(tau.norm().is_finite());
                let det = mandel_to_sym_matrix(&SVector::<f64, 6>::from_column_slice(&history[..6])).determinant();
                assert!((det - 1.0).abs() <= 1e-12, "det C_p^-1 = {}", det);
            }
            assert!(history[6] > 0.1, "not enough plastic flow: alpha = {}", history[6]);
        }
    }

    /// After plastic loading, the configuration F = R U_p with U_p = C_p^{1/2} is stress-free and
    /// reached elastically, for any rotation R.
    #[test]
    fn test_elastic_unloading_to_stress_free_state() {
        let mut rng = seeded(41);
        for _ in 0..50 {
            let f = rng.rotation() * expm_sym(&rng.symmetric(0.4));
            let mut history = HenckyMises::initial_history();
            evaluate::<2, 7, 4, 4, HenckyMises>(&f, &mut history, &MISES).unwrap();
            assert!(history[6] > 0.0);
            let cp_inv = mandel_to_sym_matrix(&SVector::<f64, 6>::from_column_slice(&history[..6]));
            // U_p = C_p^{1/2} = exp(-1/2 ln C_p^{-1})
            let (half_log_cp_inv, _, _) = half_log_spectral(&cp_inv).unwrap();
            let u_p = expm_sym(&(-half_log_cp_inv));
            let committed = history;
            let tau = evaluate::<2, 7, 4, 4, HenckyMises>(&(rng.rotation() * u_p), &mut history, &MISES)
                .unwrap()
                .tau;
            // round-off in b_tr of a few eps, times the elastic moduli
            assert!(tau.norm() <= 1e-13 * (MU + KAPPA), "residual stress {}", tau.norm());
            assert_eq!(history, committed);
        }
    }

    /// For radial loading in logarithmic strain space, the backward-Euler return mapping of von
    /// Mises plasticity with linear hardening is exact, so one step and many steps agree. No
    /// error accumulates with the number of steps.
    #[test]
    fn test_step_size_independence_radial_loading() {
        let direction = Vector3::new(1.0, -0.3, -0.4);
        let f_at = |t: f64| Matrix3::from_diagonal(&direction.map(|d| (0.8 * t * d).exp()));
        let rotation = rotation(Vector3::new(0.3, 0.2, 1.0), 0.7);
        for n_steps in [1, 7, 50] {
            let mut history_one = HenckyMises::initial_history();
            let mut history_many = HenckyMises::initial_history();
            let tau_one = evaluate::<2, 7, 4, 4, HenckyMises>(&(rotation * f_at(1.0)), &mut history_one, &MISES)
                .unwrap()
                .tau;
            let mut tau_many = Matrix3::zeros();
            for k in 1..=n_steps {
                let t = k as f64 / n_steps as f64;
                tau_many = evaluate::<2, 7, 4, 4, HenckyMises>(&(rotation * f_at(t)), &mut history_many, &MISES)
                    .unwrap()
                    .tau;
            }
            assert!((tau_one - tau_many).norm() <= 1e-10 * tau_one.norm());
            assert!((history_one[6] - history_many[6]).abs() <= 1e-10 * history_one[6]);
        }
    }

    /// The rank-3 elastic tangent equals the general tangent with D = C, with and without a
    /// precomputed eigen decomposition, for large rotations and plastic deformations.
    #[test]
    fn test_elastic_kirchhoff_tangent() {
        let mut rng = seeded(51);
        let elastic_tangent = isotropic_elastic_tangent::<6>(MU, KAPPA);
        let mut max_error: f64 = 0.0;
        for i in 0..4_000 {
            let elastic_norm = rng.log_uniform(1e-8, 0.5);
            let (f, cp_inv) = random_state(&mut rng, elastic_norm);
            let kinematics = HenckyKinematics::new(&row_major(&f), &cp_inv, i % 2 == 0).unwrap();
            let general = kinematics.kirchhoff_tangent(&elastic_tangent);
            let rank_3 = kinematics.elastic_kirchhoff_tangent(&elastic_tangent);
            max_error = max_error.max((general - rank_3).norm() / general.norm());
        }
        assert!(max_error <= 1e-14, "max. relative error {:e}", max_error);
    }
}
