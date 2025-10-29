use crate::consts::*;
use crate::plasticity::*;
use crate::interfaces::*;
use crate::mandel::*;
use crate::{create_history_parameter_struct};
use nalgebra::RowSVector;
use nalgebra::Scalar;
//use crate::impl_from_array;
use nalgebra::{SMatrix, SVector};

create_history_parameter_struct!(
    DruckerPragerHyperbolicParameters,
    6,
    6,
    [
        (mu, (QDim::Scalar)),
        (kappa, (QDim::Scalar)),
        (a, (QDim::Scalar)),
        (b, (QDim::Scalar)),
        (d, (QDim::Scalar)),
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
/// - `a`: slope of the yield surface in $I_1,\sqrt{J_2}$ space
/// - `b`: Yield strength at zero pressure
/// - `d`: Smoothing parameter
/// - `b_flow`: slope of the flow-potential, use `b_flow=b` for associated flow
#[derive(Default, Clone, Copy)]
pub struct DruckerPragerHyperbolic3D {
    parameters: DruckerPragerHyperbolicParameters,
    elastic_tangent: SMatrix<f64, 6, 6>,
    elastic_tangent_inv: SMatrix<f64, 6, 6>,
    f: f64,
    df_dsigma: RowSVector<f64, 6>,
    df_dkappa: RowSVector<f64, 1>,
    g: SVector<f64, 6>,
    dg_dkappa: SMatrix<f64, 6, 1>,
    dg_dsigma: SMatrix<f64, 6, 6>,
    k: SVector<f64, 1>,
    dk_dsigma: SMatrix<f64, 1, 6>,
    dk_dkappa: SMatrix<f64, 1, 1>,
    del_plastic_strain: SVector<f64, 6>,
}

impl Plasticity<6, 6, 6, 1> for DruckerPragerHyperbolic3D {
    type Parameters = DruckerPragerHyperbolicParameters;

    fn new(parameters: &Self::Parameters) -> Self {
        DruckerPragerHyperbolic3D {
            parameters: parameters.clone(),
            elastic_tangent: isotropic_elastic_tangent(parameters.mu, parameters.kappa),
            elastic_tangent_inv: isotropic_elastic_tangent_inv(parameters.mu, parameters.kappa),
            ..Default::default()
        }
    }

    fn set_model_state(
        &mut self,
        sigma_0: &SVector<f64, 6>,
        sigma_1: &SVector<f64, 6>,
        del_eps: &SVector<f64, 6>,
        _kappa: &SVector<f64, 1>,
    ) {
        const PROJECTION_DEV: SMatrix<f64, 6, 6> = const { projection_dev::<6>() };
        const SYM_ID: SVector<f64, 6> = const { sym_id::<6>() };
        // Implementation of setting model state
        let (i_1, s) = sigma_1.trace_dev();
        let j_2 = 0.5 * s.norm_squared();
        self.f = (j_2+self.parameters.d.powi(2)).sqrt() + self.parameters.b * i_1 - self.parameters.a;
        let df_di_1 = self.parameters.b;
        let df_dj_2 = 0.5*(j_2 + self.parameters.d.powi(2)).sqrt().recip();
        let _df_di_1i_1 = 0.0;
        let df_dj_2j_2 = -1.0/4.0*(j_2 + self.parameters.d.powi(2)).powf(-3_f64/2.0);

        let df_dsigma = df_di_1 * &SYM_ID + df_dj_2 * &s;
        self.df_dsigma = df_dsigma.transpose();
        self.g = {
            if self.parameters.b == self.parameters.b_flow {
                // associated flow
                df_dsigma
            } else {
                //non-associated flow
                self.parameters.b_flow * &SYM_ID + df_dj_2 * &s
            }
        };
        self.dg_dsigma = s * df_dj_2j_2 * s.transpose() + df_dj_2 * PROJECTION_DEV;

        self.del_plastic_strain = del_eps - self.elastic_tangent_inv * (sigma_1 - sigma_0);
        
        
        let g_norm = self.g.norm();
        self.k = SVector::from_element((2_f64/3_f64).sqrt()*g_norm);
        self.dk_dsigma = ((2_f64/3_f64).sqrt()/g_norm)* self.g.transpose() * &self.dg_dsigma;
        self.dk_dkappa = ((2_f64/3_f64).sqrt()/g_norm)* self.g.transpose() * &self.dg_dkappa;
    }

    fn f(&self) -> f64 {
        // Implementation of f function
        self.f
    }
    fn df_dsigma(&self) -> &RowSVector<f64, 6> {
        // Implementation of df_dsigma
        &self.df_dsigma
    }
    fn df_dkappa(&self) -> &RowSVector<f64, 1> {
        // Implementation of df_dkappa
        &self.df_dkappa
    }
    fn g(&self) -> &SVector<f64, 6> {
        // Implementation of g function
        &self.g
    }
    fn dg_dkappa(&self) -> &SMatrix<f64, 6, 1> {
        // Implementation of dg_dkappa
        &self.dg_dkappa
    }
    fn dg_dsigma(&self) -> &SMatrix<f64, 6, 6> {
        // Implementation of dg_dsigma
        &self.dg_dsigma
    }
    fn k(&self) -> &SVector<f64, 1> {
        // Implementation of k function
        &self.k
    }
    fn dk_dsigma(&self) -> &SMatrix<f64, 1, 6> {
        // Implementation of dk_dsigma
        &self.dk_dsigma
    }
    fn dk_dkappa(&self) -> &SMatrix<f64, 1, 1> {
        // Implementation of dk_dkappa
        &self.dk_dkappa
    }
    fn elastic_tangent(&self) -> &SMatrix<f64, 6, 6> {
        // Implementation of elastic_tangent
        &self.elastic_tangent
    }
    fn elastic_tangent_inv(&self) -> &SMatrix<f64, 6, 6> {
        // Implementation of elastic_tangent_inv
        &self.elastic_tangent_inv
    }
    fn del_plastic_strain(&self) -> &SVector<f64, 6> {
        // Implementation of del_plastic_strain
        &self.del_plastic_strain
    }
}
