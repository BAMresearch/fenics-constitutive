use crate::consts::*;
use crate::plasticity::*;
use crate::interfaces::*;
use crate::mandel::*;
use crate::{create_history_parameter_struct};
use nalgebra::Scalar;
//use crate::impl_from_array;
use nalgebra::{SMatrix, SVector};

create_history_parameter_struct!(
    DruckerPragerHyperbolicParameters,
    7,
    7,
    [
        (mu, (QDim::Scalar)),
        (kappa, (QDim::Scalar)),
        (a, (QDim::Scalar)),
        (b, (QDim::Scalar)),
        (d, (QDim::Scalar)),
        (h, (QDim::Scalar)),
        (b_flow, (QDim::Scalar))
    ]
);

/// A hyperbolically approximated Drucker-Prager plasticity model for 3D stress states.
///
/// This struct represents the Drucker-Prager yield criterion with eihter associated or non-associated flow rule.
/// The yield function is defined as: $f = \sqrt{J_2+(bd)^2} + b\cdot I_1 - a$, where:
/// - $J_2$ is the second invariant of the deviatoric stress tensor
/// - $I_1$ is the first invariant of the stress tensor
/// - $a$ and $b$ are material parameters that describe the yield surface as in the [`DruckerPrager3D`] model. $d$ is an additional smoothing parameter for the tip.
/// - $h$ is a hardening parameter that expands the yield surface by multiplying $(1+h\alpha)$ to the parameters $a,d$. The slope $b$ remains constant.ts
/// - $b_{flow}$ defines the slope of the flow rule which is equal to $b$ for associated flow. For $b=0$ the return direction is purely deviatoric (radial return algorithm)
///
///
/// This struct does not implement the stress return algorithm but implements
/// the required functions like the yield function, flow rule, etc.
/// via the [`Plasticity`] trait. It is to be used within the [`IsotropicPlasticityModel3D`]
/// in order to solve the plasticity problem.
///
/// # Parameters
/// - `mu`: Shear modulus
/// - `kappa`: Bulk modulus
/// - `b`: slope of the yield surface in $I_1,\sqrt{J_2}$ space
/// - `a`: Yield strength at zero pressure
/// - `d`: Smoothing parameter
/// - `h`: Hardening parameter
/// - `b_flow`: slope of the flow-potential, use `b_flow=b` for associated flow
#[derive(Default, Clone, Copy)]
pub struct DruckerPragerHyperbolic3D {
    parameters: DruckerPragerHyperbolicParameters,
    elastic_tangent: SMatrix<f64, 6, 6>,
    elastic_tangent_inv: SMatrix<f64, 6, 6>,
}

impl Plasticity<6, 7, 7, 1> for DruckerPragerHyperbolic3D {
    type Parameters = DruckerPragerHyperbolicParameters;

    fn new(parameters: &Self::Parameters) -> Self {
        DruckerPragerHyperbolic3D {
            parameters: parameters.clone(),
            elastic_tangent: isotropic_elastic_tangent(parameters.mu, parameters.kappa),
            elastic_tangent_inv: isotropic_elastic_tangent_inv(parameters.mu, parameters.kappa),
            ..Default::default()
        }
    }
    fn calculate_f_only(&self, sigma: &SVector<f64, 6>, kappa: &SVector<f64, 1>)->f64 {
        let (i_1, s) = sigma.trace_dev();
        let j_2 = 0.5 * s.norm_squared();

        let a = (1.0+self.parameters.h * kappa.x)*self.parameters.a;
        let d = (1.0+self.parameters.h * kappa.x)*self.parameters.d;
        let b = self.parameters.b;


        return (j_2+(d*b).powi(2)).sqrt() + b * i_1 - a;
    }
    fn set_model_state(
        &self,
        sigma: &SVector<f64, 6>,
        kappa: &SVector<f64, 1>,
        state: &mut PlasticityState<6, 7, 7, 1>,
    ) {
        const PROJECTION_DEV: SMatrix<f64, 6, 6> = const { projection_dev::<6>() };
        const SYM_ID: SVector<f64, 6> = const { sym_id::<6>() };
        // Implementation of setting model state
        let (i_1, s) = sigma.trace_dev();
        let j_2 = 0.5 * s.norm_squared();

        let a = (1.0+self.parameters.h * kappa.x)*self.parameters.a;
        let d = (1.0+self.parameters.h * kappa.x)*self.parameters.d;
        let b = self.parameters.b;
        let da_dkappa= self.parameters.h*self.parameters.a;
        let dd_dkappa= self.parameters.h*self.parameters.d;

        state.f = (j_2+(d*b).powi(2)).sqrt() + b * i_1 - a;
        let df_di_1 = b;
        let df_dj_2 = (1_f64/2.0)*(j_2 + b.powi(2)*d.powi(2)).sqrt().recip();
        let _df_di_1i_1 = 0.0;
        let df_dj_2j_2 = -1.0/4.0*(j_2 + b.powi(2)*d.powi(2)).powf(-3_f64/2.0);

        let df_dsigma = df_di_1 * &SYM_ID + df_dj_2 * &s;
        state.df_dsigma = df_dsigma.transpose();

        state.df_dkappa.x = b.powi(2)*d*dd_dkappa/(j_2 + b.powi(2)*d.powi(2)).sqrt() - da_dkappa;

        state.g = {
            if self.parameters.b == self.parameters.b_flow {
                // associated flow
                df_dsigma
            } else {
                //non-associated flow
                self.parameters.b_flow * &SYM_ID + df_dj_2 * &s
            }
        };
        state.dg_dsigma = s * df_dj_2j_2 * s.transpose() + df_dj_2 * PROJECTION_DEV;

        let df_dj_2_kappa = -1_f64/2.0*d*dd_dkappa*b.powi(2)*(j_2 + b.powi(2)*d.powi(2)).powf(-3_f64/2.0);
        state.dg_dkappa = df_dj_2_kappa * &s; //df_di_1_kappa=0

        
        let g_norm = state.g.norm();
        state.k = SVector::from_element((2_f64/3_f64).sqrt()*g_norm);
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
