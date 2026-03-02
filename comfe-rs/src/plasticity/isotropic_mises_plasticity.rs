use crate::consts::*;
//use crate::impl_array_equivalent;
use crate::interfaces::*;
use crate::mandel::*;
use crate::plasticity::*;
use crate::create_history_parameter_struct;
use nalgebra::RowSVector;
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

    fn set_model_state(
        &mut self,
        sigma_0: &SVector<f64, 6>,
        sigma_1: &SVector<f64, 6>,
        kappa: &SVector<f64, 1>,
    ) {
        const PROJECTION_DEV: SMatrix<f64, 6, 6> = const { projection_dev::<6>() };
        //const SYM_ID: SVector<f64, 6> = const { sym_id::<6>() };
        // Implementation of setting model state
        let (_i_1, s) = sigma_1.trace_dev();


        let j_2 = 0.5 * s.norm_squared();
        self.f = (3.0*j_2).sqrt() - self.parameters.y_0 - self.parameters.h * kappa.x;

        let df_dj_2 = 1.5 / (3.0*j_2).sqrt();
        let df_dj_2j_2 = -(9./4.) / (3.0*j_2).powf(3.0/2.0);

        let df_dsigma = df_dj_2 * &s;
        self.df_dsigma = df_dsigma.transpose();
        self.df_dkappa.x = - self.parameters.h;

        self.g = df_dsigma;
        // This derivative is the same for both associated and non-associated flow
        self.dg_dsigma = &s * df_dj_2j_2 * &s.transpose() + df_dj_2 * &PROJECTION_DEV;
        
        let g_norm = self.g.norm();
        self.k.x =(2_f64/3_f64).sqrt()*g_norm;
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
}
