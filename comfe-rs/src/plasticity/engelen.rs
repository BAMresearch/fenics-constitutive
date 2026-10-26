use crate::consts::*;
//use crate::impl_array_equivalent;
use crate::interfaces::*;
use crate::mandel::*;
use crate::plasticity::*;
use crate::create_history_parameter_struct;
use nalgebra::Scalar;
//, q_dim_data_type};
//use crate::impl_from_array;
use nalgebra::{SMatrix, SVector};

create_history_parameter_struct!(
    EngelenParameters,
    7,
    7,
    [
        (mu, (QDim::Scalar)),
        (kappa, (QDim::Scalar)),
        (y_0, (QDim::Scalar)),
        (h, (QDim::Scalar)),
        (alpha_0, (QDim::Scalar)),
        (e_f, (QDim::Scalar)),
        (omega_max, (QDim::Scalar))
    ]
);
/// A von Mises plasticity model with linear hardening for 3D stress states. This is a duplicate
///  implementation to test the implementation of the tangent.
///
/// This struct implements the von Mises yield criterion with linear isotropic hardening.
/// The yield function is defined as: $f = \sqrt{\frac{3}{2} s:s} - \sigma_y$, where:
/// - $s$ is the deviatoric stress tensor
/// - $\sigma_y = y_0 + h \cdot \alpha$ is the current yield stress
/// - $\alpha$ is the equivalent plastic strain
///
/// # Parameters
/// - `mu`: Shear modulus
/// - `kappa`: Bulk modulus
/// - `y_0`: Initial yield stress
/// - `h`: Linear hardening modulus
#[derive(Default, Clone, Copy)]
pub struct Engelen3D {
    parameters: EngelenParameters,
    elastic_tangent: SMatrix<f64, 6, 6>,
    elastic_tangent_inv: SMatrix<f64, 6, 6>,
    omega: f64,
    domega_dkappa_nonlocal: f64,
}

impl Plasticity<6, 7,7, 1> for Engelen3D {
    type Parameters = EngelenParameters;

    fn new(parameters: &Self::Parameters) -> Self {
        Engelen3D {
            parameters: parameters.clone(),
            elastic_tangent: isotropic_elastic_tangent(parameters.mu, parameters.kappa),
            elastic_tangent_inv: isotropic_elastic_tangent_inv(parameters.mu, parameters.kappa),
            ..Default::default()
        }
    }
    fn calculate_f_only(&self, sigma: &SVector<f64, 6>, kappa: &SVector<f64, 1>)->f64 {
        let (_i_1, s) = sigma.trace_dev();

        let j_2 = 0.5 * s.norm_squared();
        return (3.0*j_2).sqrt() - (1.0-self.omega)*(self.parameters.y_0 + self.parameters.h * kappa.x);
    }
    fn set_model_state(
        &self,
        sigma: &SVector<f64, 6>,
        kappa: &SVector<f64, 1>,
        state: &mut PlasticityState<6, 7, 7, 1>,
    ) {
        const PROJECTION_DEV: SMatrix<f64, 6, 6> = const { projection_dev::<6>() };
        //const SYM_ID: SVector<f64, 6> = const { sym_id::<6>() };
        // Implementation of setting model state
        let (_i_1, s) = sigma.trace_dev();


        let j_2 = 0.5 * s.norm_squared();
        state.f = (3.0*j_2).sqrt() - (1.0-self.omega)*(self.parameters.y_0 + self.parameters.h * kappa.x);

        let df_dj_2 = 1.5 / (3.0*j_2).sqrt();
        let df_dj_2j_2 = -(9./4.) / (3.0*j_2).powf(3.0/2.0);

        let df_dsigma = df_dj_2 * &s;
        state.df_dsigma = df_dsigma.transpose();
        state.df_dkappa.x = -(1.0-self.omega) * self.parameters.h;

        state.g = df_dsigma;
        // This derivative is the same for both associated and non-associated flow
        state.dg_dsigma = &s * df_dj_2j_2 * &s.transpose() + df_dj_2 * &PROJECTION_DEV;
        // the flow direction does not depend on kappa
        state.dg_dkappa = SMatrix::zeros();
        
        let g_norm = state.g.norm();
        state.k.x =(2_f64/3_f64).sqrt()*g_norm;
        state.dk_dsigma = ((2_f64/3_f64).sqrt()/g_norm)* state.g.transpose() * &state.dg_dsigma;
        state.dk_dkappa = ((2_f64/3_f64).sqrt()/g_norm)* state.g.transpose() * &state.dg_dkappa;
    }

    fn elastic_tangent(&self) -> &SMatrix<f64, 6, 6> {
        // Implementation of elastic_tangent
        &self.elastic_tangent
    }
    fn elastic_tangent_inv(&self) -> &SMatrix<f64, 6, 6> {
        // Implementation of elastic_tangent_inv
        &self.elastic_tangent_inv
    }
    fn is_admissible(&self, kappa: &SVector<f64, 1>) -> bool {
        // kappa is the accumulated plastic strain
        kappa.x >= 0.0
    }
}

impl GradientPlasticity<6, 7, 7, 1> for Engelen3D {
    fn set_nonlocal_state(
        &mut self,
        kappa_nonlocal: &SVector<f64, 1>,
        kappa_nonlocal_max: &SVector<f64, 1>,
    ) {
        if kappa_nonlocal_max.x >= self.parameters.alpha_0 {
            self.omega = (1.
                - f64::exp((self.parameters.alpha_0 - kappa_nonlocal_max.x) / self.parameters.e_f))
                * self.parameters.omega_max;
            // damage only grows while the current nonlocal quantity drives the maximum;
            // otherwise omega is frozen and does not change with kappa_nonlocal
            self.domega_dkappa_nonlocal = if kappa_nonlocal.x >= kappa_nonlocal_max.x {
                (self.parameters.omega_max / self.parameters.e_f)
                    * f64::exp((self.parameters.alpha_0 - kappa_nonlocal_max.x) / self.parameters.e_f)
            } else {
                0.0
            };
        } else {
            self.omega = 0.0;
            self.domega_dkappa_nonlocal = 0.0;
        }
    }

    fn set_nonlocal_derivatives(
        &self,
        _sigma: &SVector<f64, 6>,
        kappa: &SVector<f64, 1>,
        _state: &PlasticityState<6, 7, 7, 1>,
        nonlocal_state: &mut NonlocalState<6, 7, 7, 1>,
    ) {
        // f = sqrt(3 J_2) - (1 - omega) * (y_0 + h * kappa)
        // => df/dkappa_nonlocal = (y_0 + h * kappa) * domega/dkappa_nonlocal
        let domega_dkappa = self.domega_dkappa_nonlocal;

        nonlocal_state.df_dkappa_nonlocal.x =
            (self.parameters.y_0 + self.parameters.h * kappa.x) * domega_dkappa;
        // the flow direction, and therefore k, does not depend on omega
        nonlocal_state.dg_dkappa_nonlocal = SMatrix::zeros();
        nonlocal_state.dk_dkappa_nonlocal = SMatrix::zeros();
    }

    fn omega(&self) -> f64 {
        self.omega
    }
}
