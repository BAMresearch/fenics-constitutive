use crate::consts::*;
use crate::impl_array_equivalent;
use crate::interfaces::*;
use crate::mandel::*;
use crate::plasticity::*;
use crate::{create_history_parameter_struct, q_dim_data_type};
//use crate::impl_from_array;
use nalgebra::{SMatrix, SVector};

create_history_parameter_struct!(
    DruckerPragerParameters,
    5,
    5,
    [
        (mu, (QDim::Scalar)),
        (kappa, (QDim::Scalar)),
        (a, (QDim::Scalar)),
        (b, (QDim::Scalar)),
        (b_flow, (QDim::Scalar))
    ]
);
/// A classic Drucker-Prager plasticity model for 3D stress states.
///
/// This struct represents the Drucker-Prager yield criterion with associated flow rule.
/// The yield function is defined as: $f = \sqrt{J_2} + b\cdot I_1 - a$, where:
/// - $J_2$ is the second invariant of the deviatoric stress tensor
/// - $I_1$ is the first invariant of the stress tensor
/// - $a$ and $b$ are material parameters that describe the yield surface.
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
/// - `b_flow`: slope of the flow-potential, use `b_flow=b` for associated flow
#[derive(Default, Clone, Copy)]
pub struct DruckerPrager3D {
    parameters: DruckerPragerParameters,
    elastic_tangent: SMatrix<f64, 6, 6>,
    elastic_tangent_inv: SMatrix<f64, 6, 6>,
    f: f64,
    df_dsigma: SVector<f64, 6>,
    df_dkappa: SVector<f64, 1>,
    g: SVector<f64, 6>,
    dg_dkappa: SMatrix<f64, 6, 1>,
    dg_dsigma: SMatrix<f64, 6, 6>,
    k: SVector<f64, 1>,
    dk_dsigma: SMatrix<f64, 1, 6>,
    dk_dkappa: SMatrix<f64, 1, 1>,
    del_plastic_strain: SVector<f64, 6>,
}

impl Plasticity<6, 5, 5, 1> for DruckerPrager3D {
    type Parameters = DruckerPragerParameters;

    fn new(parameters: &Self::Parameters) -> Self {
        DruckerPrager3D {
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
        self.f = j_2.sqrt() + self.parameters.b * i_1 - self.parameters.a;
        let df_di_1 = self.parameters.b;
        let df_dj_2 = 0.5 / j_2.sqrt();
        let _df_di_1i_1 = 0.0;
        let df_dj_2j_2 = -0.25 / (j_2 * j_2.sqrt());

        self.df_dsigma = df_di_1 * &SYM_ID + df_dj_2 * &s;
        self.g = {
            if self.parameters.b == self.parameters.b_flow {
                // associated flow
                self.df_dsigma.clone()
            } else {
                //non-associated flow
                self.parameters.b_flow * &SYM_ID + df_dj_2 * &s
            }
        };
        // This derivative is the same for both associated and non-associated flow
        self.dg_dsigma = &s * df_dj_2j_2 * &s.transpose() + df_dj_2 * &PROJECTION_DEV;

        self.del_plastic_strain = del_eps - &self.elastic_tangent_inv * (sigma_1 - sigma_0);
        let pl_norm = self.del_plastic_strain.norm();
        self.k = SMatrix::from_element(f64::sqrt(2. / 3.) * pl_norm);
        self.dk_dsigma = {
            if pl_norm == 0.0 {
                SMatrix::zeros()
            } else {
                (-f64::sqrt(2. / 3.) * &self.elastic_tangent_inv * &self.del_plastic_strain / pl_norm)
                    .transpose()
            }
        };
    }

    fn f(&self) -> f64 {
        // Implementation of f function
        self.f
    }
    fn df_dsigma(&self) -> &SVector<f64, 6> {
        // Implementation of df_dsigma
        &self.df_dsigma
    }
    fn df_dkappa(&self) -> &SVector<f64, 1> {
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
