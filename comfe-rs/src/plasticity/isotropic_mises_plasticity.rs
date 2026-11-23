use crate::consts::*;
//use crate::impl_array_equivalent;
use crate::interfaces::*;
use crate::mandel::*;
use crate::plasticity::*;
use crate::create_history_parameter_struct;
//, q_dim_data_type};
//use crate::impl_from_array;
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
/// # Parameters
/// - `mu`: Shear modulus
/// - `kappa`: Bulk modulus
/// - `y_0`: Initial yield stress
/// - `h`: Linear hardening modulus
#[derive(Default, Clone, Copy)]
pub struct IsotropicMises3D {
    parameters: IsotropicMisesParameters,
}

impl Plasticity<6, 4, 4, 1> for IsotropicMises3D {
    type Parameters = IsotropicMisesParameters;

    fn new(parameters: &Self::Parameters) -> Self {
        IsotropicMises3D {
            parameters: parameters.clone(),
        }
    }
    fn calculate_f_only(&self, sigma: &SVector<f64, 6>, kappa: &SVector<f64, 1>)->f64 {
        let (_i_1, s) = sigma.trace_dev();


        let j_2 = 0.5 * s.norm_squared();
        return (3.0*j_2).sqrt() - (self.parameters.y_0 + self.parameters.h * kappa.x);
    }
    fn set_model_state(
        &self,
        sigma: &SVector<f64, 6>,
        kappa: &SVector<f64, 1>,
        state: &mut PlasticityState<6, 4, 4, 1>,
    ) {
        const PROJECTION_DEV: SMatrix<f64, 6, 6> = const { projection_dev::<6>() };
        //const SYM_ID: SVector<f64, 6> = const { sym_id::<6>() };
        // Implementation of setting model state
        let (_i_1, s) = sigma.trace_dev();


        let j_2 = 0.5 * s.norm_squared();
        state.f = (3.0*j_2).sqrt() - (self.parameters.y_0 + self.parameters.h * kappa.x);

        let df_dj_2 = 1.5 / (3.0*j_2).sqrt();
        let df_dj_2j_2 = -(9./4.) / (3.0*j_2).powf(3.0/2.0);

        let df_dsigma = df_dj_2 * &s;
        state.df_dsigma = df_dsigma.transpose();
        state.df_dkappa.x = -self.parameters.h;

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
}
