use crate::consts::*;
use crate::create_history_parameter_struct;
use crate::interfaces::*;
use crate::mandel::*;
use crate::plasticity::*;
use nalgebra::{SMatrix, SVector};

create_history_parameter_struct!(
    DPHDamageParameters,
    13,
    13,
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
        (alpha_0, (QDim::Scalar)),
        (omega_max, (QDim::Scalar))
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
    // damage and its derivative are set once per evaluation by `set_nonlocal_state`
    omega: f64,
    domega_dkappa_nonlocal: f64,
}

/// The parameters $a$, $b$, $d$ of the yield surface, or their derivatives with
/// respect to a scalar quantity.
#[derive(Clone, Copy)]
struct YieldParameters {
    a: f64,
    b: f64,
    d: f64,
}

impl DPHDamage3D {
    /// Interpolates between the hardened yield surface and the residual surface:
    /// $x = (1 + h\kappa)(1 - \omega) x_y + \omega x_r$ for $x \in \{a, b, d\}$.
    fn yield_parameters(&self, kappa: f64, omega: f64) -> YieldParameters {
        let p = &self.parameters;
        let factor = (1.0 + p.h * kappa) * (1.0 - omega);
        YieldParameters {
            a: factor * p.a_y + omega * p.a_r,
            b: factor * p.b_y + omega * p.b_r,
            d: factor * p.d_y + omega * p.d_r,
        }
    }

    fn dyield_parameters_dkappa(&self, omega: f64) -> YieldParameters {
        let p = &self.parameters;
        let factor = p.h * (1.0 - omega);
        YieldParameters {
            a: factor * p.a_y,
            b: factor * p.b_y,
            d: factor * p.d_y,
        }
    }

    fn dyield_parameters_domega(&self, kappa: f64) -> YieldParameters {
        let p = &self.parameters;
        let factor = -(1.0 + p.h * kappa);
        YieldParameters {
            a: factor * p.a_y + p.a_r,
            b: factor * p.b_y + p.b_r,
            d: factor * p.d_y + p.d_r,
        }
    }
}

/// $f = I_1 + a\sqrt{J_2 + b^2}/b - d$
fn yield_function(i_1: f64, j_2: f64, y: YieldParameters) -> f64 {
    i_1 + y.a * (j_2 + y.b.powi(2)).sqrt() / y.b - y.d
}

/// Derivative of the yield function with respect to a scalar quantity $t$,
/// given the derivatives `dy` of $a$, $b$, $d$ with respect to $t$.
fn df_dparameter(j_2: f64, y: YieldParameters, dy: YieldParameters) -> f64 {
    let (a, b) = (y.a, y.b);
    -(j_2 + b.powi(2)).sqrt() * a * dy.b / b.powi(2)
        + (j_2 + b.powi(2)).sqrt() * dy.a / b
        - dy.d
        + a * dy.b / (j_2 + b.powi(2)).sqrt()
}

/// Derivative of $\partial f/\partial J_2$ with respect to a scalar quantity $t$,
/// given the derivatives `dy` of $a$, $b$, $d$ with respect to $t$.
fn ddf_dj_2_dparameter(j_2: f64, y: YieldParameters, dy: YieldParameters) -> f64 {
    let (a, b) = (y.a, y.b);
    -1.0 / 2.0 * a * dy.b / ((j_2 + b.powi(2)).sqrt() * b.powi(2))
        + (1.0 / 2.0) * dy.a / ((j_2 + b.powi(2)).sqrt() * b)
        - 1.0 / 2.0 * a * dy.b / (j_2 + b.powi(2)).powf(3.0 / 2.0)
}

impl Plasticity<6, 13, 13, 1> for DPHDamage3D {
    type Parameters = DPHDamageParameters;

    fn new(parameters: &Self::Parameters) -> Self {
        DPHDamage3D {
            parameters: parameters.clone(),
            elastic_tangent: isotropic_elastic_tangent(parameters.mu, parameters.kappa),
            elastic_tangent_inv: isotropic_elastic_tangent_inv(parameters.mu, parameters.kappa),
            ..Default::default()
        }
    }
    fn calculate_f_only(&self, sigma: &SVector<f64, 6>, kappa: &SVector<f64, 1>) ->f64{
        let (i_1, s) = sigma.trace_dev();
        let j_2 = 0.5 * s.norm_squared();
        yield_function(i_1, j_2, self.yield_parameters(kappa.x, self.omega))
    }
    fn set_model_state(
        &self,
        sigma: &SVector<f64, 6>,
        kappa: &SVector<f64, 1>,
        state: &mut PlasticityState<6, 13, 13, 1>,
    ) {
        const PROJECTION_DEV: SMatrix<f64, 6, 6> = const { projection_dev::<6>() };
        const SYM_ID: SVector<f64, 6> = const { sym_id::<6>() };
        let (i_1, s) = sigma.trace_dev();
        let j_2 = 0.5 * s.norm_squared();

        let y = self.yield_parameters(kappa.x, self.omega);
        let dy_dkappa = self.dyield_parameters_dkappa(self.omega);
        let (a, b) = (y.a, y.b);

        state.f = yield_function(i_1, j_2, y);
        assert!(!state.f.is_nan(), "f is NaN");
        assert!(!state.f.is_infinite(), "f is infinite");

        // df_di_1 = 1
        let df_dj_2 = (1.0 / 2.0) * a / ((j_2 + b.powi(2)).sqrt() * b);

        state.df_dkappa.x = df_dparameter(j_2, y, dy_dkappa);

        state.df_dsigma = (&SYM_ID + df_dj_2 * &s).transpose();
        state.g = (1.0 - self.parameters.radial_factor) * &SYM_ID + df_dj_2 * &s;
        let df_dj_2j_2 =
            -1_f64 / 4.0 * a / ((j_2 + b.powi(2)).powf(3_f64 / 2.0) * b);
        state.dg_dsigma = &s * df_dj_2j_2 * &s.transpose() + df_dj_2 * &PROJECTION_DEV;
        state.dg_dkappa = ddf_dj_2_dparameter(j_2, y, dy_dkappa) * &s;

        let g_norm = state.g.norm();
        state.k.x = (2_f64 / 3_f64).sqrt() * g_norm;
        state.dk_dsigma = ((2_f64 / 3_f64).sqrt() / g_norm) * state.g.transpose() * &state.dg_dsigma;
        state.dk_dkappa = ((2_f64 / 3_f64).sqrt() / g_norm) * state.g.transpose() * &state.dg_dkappa;
    }

    fn elastic_tangent(&self) -> &SMatrix<f64, 6, 6> {
        // Implementation of elastic_tangent
        &self.elastic_tangent
    }
    fn elastic_tangent_inv(&self) -> &SMatrix<f64, 6, 6> {
        // Implementation of elastic_tangent_inv
        &self.elastic_tangent_inv
    }
    fn characteristic_stress(&self, sigma_tr: &SVector<f64, 6>) -> f64 {
        // von Mises stress on the undamaged, unhardened yield surface at i_1 = 0
        let y = self.yield_parameters(0.0, 0.0);
        let j_2 = (y.d * y.b / y.a).powi(2) - y.b.powi(2);
        if j_2 > 0.0 {
            (3.0 * j_2).sqrt()
        } else {
            sigma_tr.norm().max(f64::MIN_POSITIVE)
        }
    }
    fn is_admissible(&self, kappa: &SVector<f64, 1>) -> bool {
        // kappa < -1/h makes the hardening factor (1 + h*kappa) negative, which crosses a
        // singularity of the yield function (b -> 0) where Newton stagnates
        kappa.x >= 0.0
    }
}

impl GradientPlasticity<6, 13, 13, 1> for DPHDamage3D {
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
        sigma: &SVector<f64, 6>,
        kappa: &SVector<f64, 1>,
        state: &PlasticityState<6, 13, 13, 1>,
        nonlocal_state: &mut NonlocalState<6, 13, 13, 1>,
    ) {
        let (_i_1, s) = sigma.trace_dev();
        let j_2 = 0.5 * s.norm_squared();

        // chain rule: d(a, b, d)/dkappa_nonlocal = d(a, b, d)/domega * domega/dkappa_nonlocal
        let y = self.yield_parameters(kappa.x, self.omega);
        let dy_domega = self.dyield_parameters_domega(kappa.x);
        let domega = self.domega_dkappa_nonlocal;
        let dy_dkappa_nonlocal = YieldParameters {
            a: dy_domega.a * domega,
            b: dy_domega.b * domega,
            d: dy_domega.d * domega,
        };

        nonlocal_state.df_dkappa_nonlocal.x = df_dparameter(j_2, y, dy_dkappa_nonlocal);
        nonlocal_state.dg_dkappa_nonlocal = ddf_dj_2_dparameter(j_2, y, dy_dkappa_nonlocal) * &s;

        let g_norm = state.g.norm();
        nonlocal_state.dk_dkappa_nonlocal = ((2_f64 / 3_f64).sqrt() / g_norm)
            * state.g.transpose()
            * &nonlocal_state.dg_dkappa_nonlocal;
    }

    fn omega(&self) -> f64 {
        self.omega
    }
}
