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
    elastic_tangent: SMatrix<f64, 6, 6>,
    elastic_tangent_inv: SMatrix<f64, 6, 6>,
}

impl Plasticity<6, 4, 4, 1> for IsotropicMises3D {
    type Parameters = IsotropicMisesParameters;

    fn new(parameters: &Self::Parameters) -> Self {
        IsotropicMises3D {
            parameters: parameters.clone(),
            elastic_tangent: isotropic_elastic_tangent(parameters.mu, parameters.kappa),
            elastic_tangent_inv: isotropic_elastic_tangent_inv(parameters.mu, parameters.kappa),
            ..Default::default()
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

    fn elastic_tangent(&self) -> &SMatrix<f64, 6, 6> {
        // Implementation of elastic_tangent
        &self.elastic_tangent
    }
    fn elastic_tangent_inv(&self) -> &SMatrix<f64, 6, 6> {
        // Implementation of elastic_tangent_inv
        &self.elastic_tangent_inv
    }
}

// impl GradientPlasticity<6, 4, 13, 1> for IsotropicMises3D {
//     fn set_nonlocal_state(
//         &mut self,
//         kappa_nonlocal: &SVector<f64, 1>,
//         kappa_nonlocal_max: &SVector<f64, 1>,
//     ) {
//         if kappa_nonlocal_max.x >= self.parameters.alpha_0 {
//             self.omega = (1.
//                 - f64::exp((self.parameters.alpha_0 - kappa_nonlocal_max.x) / self.parameters.e_f))
//                 * self.parameters.omega_max;
//             self.domega_dkappa_nonlocal = (self.parameters.omega_max / self.parameters.e_f)
//                 * f64::exp((self.parameters.alpha_0 - kappa_nonlocal_max.x) / self.parameters.e_f);
//         } else {
//             self.omega = 0.0;
//             self.domega_dkappa_nonlocal = 0.0;
//         }

//         //TODO
//     }

//     fn set_nonlocal_derivatives(&mut self, sigma: &SVector<f64, 6>, kappa: &SVector<f64, 1>) {
//         //const PROJECTION_DEV: SMatrix<f64, 6, 6> = const { projection_dev::<6>() };
//         //const SYM_ID: SVector<f64, 6> = const { sym_id::<6>() };
//         // Implementation of setting model state
//         let (_i_1, s) = sigma.trace_dev();
//         let j_2 = 0.5 * s.norm_squared();

//         let b = (1.0 + self.parameters.h * kappa.x) * (1.0 - self.omega) * self.parameters.b_y
//             + self.omega * self.parameters.b_r;
//         let a = (1.0 + self.parameters.h * kappa.x) * (1.0 - self.omega) * self.parameters.a_y
//             + self.omega * self.parameters.a_r;

//         let domega_dkappa = self.domega_dkappa_nonlocal;

//         let db_dkappa_nonlocal =
//             -(1.0 + self.parameters.h * kappa.x) * domega_dkappa * self.parameters.b_y
//                 + domega_dkappa * self.parameters.b_r;
//         let da_dkappa_nonlocal =
//             -(1.0 + self.parameters.h * kappa.x) * domega_dkappa * self.parameters.a_y
//                 + domega_dkappa * self.parameters.a_r;
//         let dd_dkappa_nonlocal =
//             -(1.0 + self.parameters.h * kappa.x) * domega_dkappa * self.parameters.d_y
//                 + domega_dkappa * self.parameters.d_r;

//         self.df_dkappa_nonlocal.x = -(j_2 + b.powi(2)).sqrt() * a * db_dkappa_nonlocal / b.powi(2)
//             + (j_2 + b.powi(2)).sqrt() * da_dkappa_nonlocal / b
//             - dd_dkappa_nonlocal
//             + a * db_dkappa_nonlocal / (j_2 + b.powi(2)).sqrt();

//         let df_dj_2_dkappa_nonlocal =
//             -1_f64 / 2.0 * a * db_dkappa_nonlocal * (j_2 + b.powi(2)).sqrt().recip() * b.powi(-2)
//                 + (1_f64 / 2.0) * da_dkappa_nonlocal * (j_2 + b.powi(2)).sqrt().recip() * b.recip()
//                 - 1_f64 / 2.0 * a * db_dkappa_nonlocal * (j_2 + b.powi(2)).powf(-3_f64 / 2.0);
//         self.dg_dkappa_nonlocal = df_dj_2_dkappa_nonlocal * &s;

//         let g_norm = self.g.norm();
//         self.dk_dkappa_nonlocal =
//             ((2_f64 / 3_f64).sqrt() / g_norm) * self.g.transpose() * &self.dg_dkappa_nonlocal;
//     }

//     fn df_dkappa_nonlocal(&self) -> &RowSVector<f64, 1> {
//         &self.df_dkappa_nonlocal
//     }

//     fn dg_dkappa_nonlocal(&self) -> &SMatrix<f64, 6, 1> {
//         &self.dg_dkappa_nonlocal
//     }

//     fn dk_dkappa_nonlocal(&self) -> &SMatrix<f64, 1, 1> {
//         &self.dk_dkappa_nonlocal
//     }
// }
