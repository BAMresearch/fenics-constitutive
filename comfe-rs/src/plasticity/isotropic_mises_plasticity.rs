use crate::consts::*;
use crate::interfaces::*;
use crate::mandel::*;
use crate::plasticity::*;
use crate::create_history_parameter_struct;
use nalgebra::{SMatrix, SVector};

create_history_parameter_struct!(
    IsotropicMisesParameters,
    4,
    4,
    [
        (mu, (QDim::Scalar)),
        (kappa, (QDim::Scalar)),
        (y_0, (QDim::Scalar)),
        (h, (QDim::Scalar))
    ]
);
crate::impl_isotropic_elastic_parameters!(IsotropicMisesParameters);
/// A von Mises plasticity model with linear hardening for 3D stress states. This is a duplicate
///  implementation to test the implementation of the tangent.
///
/// This struct implements the von Mises yield criterion with linear isotropic hardening.
/// The yield function is defined as: $f = \sqrt{\frac{3}{2} s:s} - \sigma_y$, where:
/// - $s$ is the deviatoric stress tensor
/// - $\sigma_y = y_0 + h \cdot \alpha$ is the current yield stress
/// - $\alpha$ is the equivalent plastic strain
///
/// # Yield surface
/// Meridian section of the yield surface ($I_1$ vs. $\sqrt{J_2}$); the deviatoric section is a
/// circle. The yield surface does not depend on $I_1$, and the yield stress $y_0 + h\alpha$ moves
/// it parallel to the $I_1$ axis.
///
#[doc = "<div>"]
#[doc = include_str!("../../../docs/figures/yield_surface_mises.svg")]
#[doc = "</div>"]
///
/// # Parameters
/// - `mu`: Shear modulus
/// - `kappa`: Bulk modulus
/// - `y_0`: Initial yield stress
/// - `h`: Linear hardening modulus
#[derive(Default, Clone, Copy)]
pub struct IsotropicMises3D {
    parameters: IsotropicMisesParameters,
}

impl IsotropicMises3D {
    fn yield_function(&self, j_2: f64, kappa: f64) -> f64 {
        (3.0 * j_2).sqrt() - (self.parameters.y_0 + self.parameters.h * kappa)
    }
}

impl Plasticity<6, 4, 4, 1> for IsotropicMises3D {
    type Parameters = IsotropicMisesParameters;

    fn new(parameters: &Self::Parameters) -> Self {
        IsotropicMises3D {
            parameters: parameters.clone(),
        }
    }
    /// The initial yield stress $y_0$.
    fn characteristic_stress(&self, _sigma_tr: &SVector<f64, 6>) -> f64 {
        self.parameters.y_0
    }
    fn calculate_f_only(&self, sigma: &SVector<f64, 6>, kappa: &SVector<f64, 1>) -> f64 {
        let (_i_1, s) = sigma.trace_dev();
        self.yield_function(0.5 * s.norm_squared(), kappa.x)
    }
    fn set_model_state(
        &self,
        sigma: &SVector<f64, 6>,
        kappa: &SVector<f64, 1>,
        state: &mut PlasticityState<6, 1>,
    ) {
        const PROJECTION_DEV: SMatrix<f64, 6, 6> = const { projection_dev::<6>() };
        let (_i_1, s) = sigma.trace_dev();


        let j_2 = 0.5 * s.norm_squared();
        state.f = self.yield_function(j_2, kappa.x);

        let df_dj_2 = 1.5 / (3.0*j_2).sqrt();
        let df_dj_2j_2 = -(9./4.) / (3.0*j_2).powf(3.0/2.0);

        let df_dsigma = df_dj_2 * &s;
        state.df_dsigma = df_dsigma.transpose();
        state.df_dkappa.x = -self.parameters.h;

        state.g = df_dsigma;
        state.dg_dsigma = &s * df_dj_2j_2 * &s.transpose() + df_dj_2 * &PROJECTION_DEV;
        // the flow direction does not depend on kappa
        state.dg_dkappa = SMatrix::zeros();
        
        let g_norm = state.g.norm();
        state.k.x =(2_f64/3_f64).sqrt()*g_norm;
        state.dk_dsigma = ((2_f64/3_f64).sqrt()/g_norm)* state.g.transpose() * &state.dg_dsigma;
        state.dk_dkappa = ((2_f64/3_f64).sqrt()/g_norm)* state.g.transpose() * &state.dg_dkappa;
    }
}
