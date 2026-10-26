use crate::consts::*;
use crate::interfaces::*;
use crate::mandel::*;
use crate::{create_history_parameter_struct};
use nalgebra::{SMatrix,  SVector, SVectorView, SVectorViewMut};

//const _: () = assert!(check_constitutive_model_maps::<
//    6,
//    2,
//    7,
//    4,
//    4,
//    MisesPlasticity3D,
//>());

/// A von Mises plasticity model with linear hardening for 3D stress states.
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
#[repr(C)]
pub struct EngelenAnalytical3D();

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
create_history_parameter_struct!(
    MisesPlasticityHistory,
    2,
    7,
    [
        (alpha_nonlocal_max, (QDim::Scalar)),
        (plastic_strain, (QDim::RotatableVector(6)))
    ]
);

impl GradientConstitutiveModelFn<6, 2, 7, 7, 7> for EngelenAnalytical3D {
    type History = MisesPlasticityHistory;
    type Parameters = EngelenParameters;

    #[inline]
    fn evaluate(
        _time: f64,
        _del_time: f64,
        del_strain: &[f64; 6],
        nonlocal_quantity: &[f64; 1],
        stress: &mut [f64; 6],
        local_quantity: &mut [f64; 1],
        tangents: Option<&mut NonlocalTangents<6>>,
        history: &mut [f64; 7],
        parameters: &[f64; 7],

    ) {
        let mises_parameters = Self::Parameters::from_array(parameters);
        let mu = mises_parameters.mu;
        let kappa = mises_parameters.kappa;
        let y_0 = mises_parameters.y_0;
        let h = mises_parameters.h;
        let alpha_0 = mises_parameters.alpha_0;
        let e_f = mises_parameters.e_f;
        let omega_max = mises_parameters.omega_max;

        const SYM_ID: SMatrix<f64, 6, 1> = const { sym_id::<6>() };
        const SYM_ID_OUTER_SYM_ID: SMatrix<f64, 6, 6> = const { sym_id_outer_sym_id::<6>() };
        const PROJECTION_DEV: SMatrix<f64, 6, 6> = const { projection_dev::<6>() };
        let elastic_tangent = isotropic_elastic_tangent(mu, kappa);
        // Unpack history
        let history_ = Self::History::from_array_mut(history);
        let alpha = local_quantity[0];

        let del_strain_vec = SVectorView::<f64, 6>::from_array(del_strain);
        let mut stress_vec = SVectorViewMut::<f64, 6>::from_array(stress);

        let (p_0, s_0) = stress_vec.vol_dev();
        let (eps_trace, eps_dev) = del_strain_vec.trace_dev();
        let p_1 = p_0 + kappa * eps_trace;

        let s_tr = &s_0 + (2. * mu) * &eps_dev;
        let s_tr_eq = s_tr.mises_norm();
        let nonlocal_max = f64::max(nonlocal_quantity[0], history_.alpha_nonlocal_max);
        history_.alpha_nonlocal_max = nonlocal_max;

        let omega = {
            if nonlocal_max >= alpha_0 {
                (1. - f64::exp(
                    (alpha_0 - nonlocal_max) / e_f,
                )) * omega_max
            } else {
                0.0
            }
        };

        // damage only grows while the current nonlocal quantity drives the maximum;
        // otherwise omega is frozen and does not change with the nonlocal quantity
        let domega_dkappa_nonlocal = {
            if nonlocal_max >= alpha_0 && nonlocal_quantity[0] >= nonlocal_max {
                (omega_max / e_f)
                    * f64::exp(
                        (alpha_0 - nonlocal_max) / e_f,
                    )
            } else {
                0.0
            }
        };

        let sigma_y = (y_0 + h * alpha) * (1.0-omega);

        //the .max(0.0) contains the check if the stress is already above the yield surface
        if s_tr_eq < sigma_y {
            // Elastic step
            stress_vec.copy_from(&(p_1 * SYM_ID + s_tr));
            if let Some(tangent) = tangents {
                tangent.dsigma_deps = elastic_tangent;
                tangent.dsigma_dnonlocal *= 0.0;
                tangent.dlocal_deps *= 0.0;
                tangent.dlocal_dnonlocal *= 0.0;
            }
            return;
        } else {
            // yield condition at the end of the step:
            // s_tr_eq - 3 mu del_alpha = (1-omega) * (y_0 + h * (alpha + del_alpha))
            let hardening = (1.0 - omega) * h;
            let del_alpha = (s_tr_eq - sigma_y) / (3. * mu + hardening);
            let del_gamma = f64::sqrt(3. / 2.) * del_alpha;
            
            let s_tr_norm = s_tr_eq * (2_f64/3_f64).sqrt();
            let theta = 1. - (2. * mu * del_gamma) / (s_tr_norm);

            // Update the equivalent plastic strain
            // determine the plastic strain
            let n = s_tr / s_tr_norm;
            history_.plastic_strain += del_gamma * &n;
            local_quantity[0] += del_alpha;

            stress_vec.copy_from(&(p_1 * &SYM_ID + &s_tr - (2.0*mu*del_gamma)*&n));

            if let Some(tangent) = tangents {
                // del_alpha = (s_tr_eq - (1-omega)(y_0 + h alpha)) / (3 mu + (1-omega) h)
                // => d(del_alpha)/domega = ((y_0 + h alpha) + h del_alpha) / (3 mu + (1-omega) h)
                let dalpha_dnonlocal = domega_dkappa_nonlocal * ((y_0 + h * alpha) + h * del_alpha)
                    / (3. * mu + hardening);

                tangent.dlocal_dnonlocal = dalpha_dnonlocal;

                let dalpha_deps =
                    f64::sqrt(3. / 2.) / (3.0 * mu + hardening) * n.transpose() * elastic_tangent;
                tangent.dlocal_deps = dalpha_deps;

                let theta_bar = 1.0 / (1.0 + (hardening / (3.0 * mu))) - (1.0 - theta);
                let tangent_new = kappa * &SYM_ID_OUTER_SYM_ID
                    + (2.0 * mu * theta) * &PROJECTION_DEV
                    - (2.0 * mu * theta_bar) * &n * &n.transpose();
                // Copy the tangent matrix to the output
                tangent.dsigma_deps = tangent_new;

                let dsigma_dnonlocal = -2.0*mu * f64::sqrt(3.0/2.0) * dalpha_dnonlocal * n;
                tangent.dsigma_dnonlocal = dsigma_dnonlocal;
            }
        }
    }
}
