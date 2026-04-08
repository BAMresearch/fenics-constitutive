use crate::consts::*;
use crate::create_history_parameter_struct;
use crate::interfaces::*;
use crate::mandel::*;
use crate::plasticity::*;
use nalgebra::RowSVector;
use nalgebra::Scalar;
//use crate::impl_from_array;
use nalgebra::{SMatrix, SVector};

create_history_parameter_struct!(
    DPHDamageParameters,
    12,
    12,
    [
        (mu, (QDim::Scalar)),
        (kappa, (QDim::Scalar)),
        (a_y, (QDim::Scalar)),
        (b_y, (QDim::Scalar)),
        (d_y, (QDim::Scalar)),
        (a_r, (QDim::Scalar)),
        (b_r, (QDim::Scalar)),
        (d_r, (QDim::Scalar)),
        (e_f, (QDim::Scalar)),
        (h, (QDim::Scalar)),
        (radial_factor, (QDim::Scalar)),
        (alpha_0, (QDim::Scalar))
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
pub struct DPHDamage3D {
    parameters: DPHDamageParameters,
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
    omega: f64,
    df_dkappa_nonlocal: RowSVector<f64, 1>,
    dg_dkappa_nonlocal: SMatrix<f64, 6, 1>,
    dk_dkappa_nonlocal: SMatrix<f64, 1, 1>,
    domega_dkappa_nonlocal: f64,
}

impl Plasticity<6, 12, 12, 1> for DPHDamage3D {
    type Parameters = DPHDamageParameters;

    fn new(parameters: &Self::Parameters) -> Self {
        DPHDamage3D {
            parameters: parameters.clone(),
            elastic_tangent: isotropic_elastic_tangent(parameters.mu, parameters.kappa),
            elastic_tangent_inv: isotropic_elastic_tangent_inv(parameters.mu, parameters.kappa),
            ..Default::default()
        }
    }

    fn set_model_state(&mut self, sigma: &SVector<f64, 6>, kappa: &SVector<f64, 1>) {
        const PROJECTION_DEV: SMatrix<f64, 6, 6> = const { projection_dev::<6>() };
        const SYM_ID: SVector<f64, 6> = const { sym_id::<6>() };
        // Implementation of setting model state
        let (i_1, s) = sigma.trace_dev();
        let j_2 = 0.5 * s.norm_squared();

        let b = (1.0 + self.parameters.h * kappa.x) * (1.0 - self.omega) * self.parameters.b_y
            + self.omega * self.parameters.b_r;
        let a = (1.0 + self.parameters.h * kappa.x) * (1.0 - self.omega) * self.parameters.a_y
            + self.omega * self.parameters.a_r;
        let d = (1.0 + self.parameters.h * kappa.x) * (1.0 - self.omega) * self.parameters.d_y
            + self.omega * self.parameters.d_r;
        let db_dkappa = self.parameters.h * (1.0 - self.omega) * self.parameters.b_y;
        let da_dkappa = self.parameters.h * (1.0 - self.omega) * self.parameters.a_y;
        let dd_dkappa = self.parameters.h * (1.0 - self.omega) * self.parameters.d_y;

        self.f = i_1 + a * (j_2 + b.powi(2)).sqrt() / b - d;
        assert!(!self.f.is_nan(), "f is NaN");
        assert!(!self.f.is_infinite(), "f is infinite");

        let df_di_1 = 1.0;

        let df_dj_2 = (1.0 / 2.0) * a / ((j_2 + b.powi(2)).sqrt() * b);

        self.df_dkappa.x = -(j_2 + b.powi(2)).sqrt() * a * db_dkappa / b.powi(2)
            + (j_2 + b.powi(2)).sqrt() * da_dkappa / b
            - dd_dkappa
            + a * db_dkappa / (j_2 + b.powi(2)).sqrt();

        self.df_dsigma = (&SYM_ID + df_dj_2 * &s).transpose();
        self.g = (1.0 - self.parameters.radial_factor) * &SYM_ID + df_dj_2 * &s;
        let df_di_1i_1 = 0.0;
        let df_dj_2j_2 =
            -1_f64 / 4.0 * self.parameters.a_y / ((j_2 + b.powi(2)).powf(3_f64 / 2.0) * b);
        self.dg_dsigma = &s * df_dj_2j_2 * &s.transpose() + df_dj_2 * &PROJECTION_DEV;
        let df_dj_2kappa = -1.0 / 2.0 * a * db_dkappa / ((j_2 + b.powi(2)).sqrt() * b.powi(2))
            + (1.0 / 2.0) * da_dkappa / ((j_2 + b.powi(2)).sqrt() * b)
            - 1.0 / 2.0 * a * db_dkappa / (j_2 + b.powi(2)).powf(3.0 / 2.0);
        self.dg_dkappa = df_dj_2kappa * &s;

        //let pl_norm = self.state.del_plastic_strain.norm();

        let g_norm = self.g.norm();
        self.k.x = (2_f64 / 3_f64).sqrt() * g_norm;
        self.dk_dsigma = ((2_f64 / 3_f64).sqrt() / g_norm) * self.g.transpose() * &self.dg_dsigma;
        self.dk_dkappa = ((2_f64 / 3_f64).sqrt() / g_norm) * self.g.transpose() * &self.dg_dkappa;
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

impl GradientPlasticity<6, 12, 12, 1> for DPHDamage3D {
    fn set_nonlocal_state(
        &mut self,
        kappa_nonlocal: &SVector<f64, 1>,
        kappa_nonlocal_max: &SVector<f64, 1>,
    ) {
        self.omega =
            1. - f64::exp((self.parameters.alpha_0 - kappa_nonlocal_max.x) / self.parameters.e_f);

        //TODO
    }

    fn set_nonlocal_derivatives(&mut self, sigma: &SVector<f64, 6>, kappa: &SVector<f64, 1>) {
        //const PROJECTION_DEV: SMatrix<f64, 6, 6> = const { projection_dev::<6>() };
        //const SYM_ID: SVector<f64, 6> = const { sym_id::<6>() };
        // Implementation of setting model state
        let (_i_1, s) = sigma.trace_dev();
        let j_2 = 0.5 * s.norm_squared();

        let b = (1.0 + self.parameters.h * kappa.x) * (1.0 - self.omega) * self.parameters.b_y
            + self.omega * self.parameters.b_r;
        let a = (1.0 + self.parameters.h * kappa.x) * (1.0 - self.omega) * self.parameters.a_y
            + self.omega * self.parameters.a_r;
        //let d = (1.0 + self.parameters.h * kappa.x) * (1.0 - self.omega) * self.parameters.d_y
        //    + self.omega * self.parameters.d_r;
        let domega_dkappa = self.domega_dkappa_nonlocal;

        let db_dkappa_nonlocal = - (1.0 + self.parameters.h * kappa.x) *domega_dkappa* self.parameters.b_y
            + domega_dkappa * self.parameters.b_r ;
        let da_dkappa_nonlocal =- (1.0 + self.parameters.h * kappa.x) *domega_dkappa* self.parameters.a_y
            + domega_dkappa * self.parameters.a_r ;
        let dd_dkappa_nonlocal = - (1.0 + self.parameters.h * kappa.x) *domega_dkappa* self.parameters.d_y
            + domega_dkappa * self.parameters.d_r;

        self.df_dkappa_nonlocal.x = -(j_2 + b.powi(2)).sqrt() * a * db_dkappa_nonlocal / b.powi(2)
            + (j_2 + b.powi(2)).sqrt() * da_dkappa_nonlocal / b
            - dd_dkappa_nonlocal
            + a * db_dkappa_nonlocal / (j_2 + b.powi(2)).sqrt();

        let df_dj_2_dkappa_nonlocal =
            -1_f64 / 2.0 * a * db_dkappa_nonlocal * (j_2 + b.powi(2)).sqrt().recip() * b.powi(-2)
                + (1_f64 / 2.0) * da_dkappa_nonlocal * (j_2 + b.powi(2)).sqrt().recip() * b.recip()
                - 1_f64 / 2.0 * a * db_dkappa_nonlocal * (j_2 + b.powi(2)).powf(-3_f64 / 2.0);
        self.dg_dkappa_nonlocal = df_dj_2_dkappa_nonlocal * &s;
        
        let g_norm = self.g.norm();
        self.dk_dkappa_nonlocal = ((2_f64 / 3_f64).sqrt() / g_norm) * self.g.transpose() * &self.dg_dkappa_nonlocal;
        
    }

    fn df_dkappa_nonlocal(&self) -> &RowSVector<f64, 1> {
        &self.df_dkappa_nonlocal
    }

    fn dg_dkappa_nonlocal(&self) -> &SMatrix<f64, 6, 1> {
        &self.dg_dkappa_nonlocal
    }

    fn dk_dkappa_nonlocal(&self) -> &SMatrix<f64, 1, 1> {
        &self.dk_dkappa_nonlocal
    }
}
