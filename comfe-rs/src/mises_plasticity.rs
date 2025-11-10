use crate::consts::*;
use crate::interfaces::*;
use crate::mandel::*;
use crate::{create_history_parameter_struct};
use nalgebra::{SMatrix,  SVector, SVectorView, SVectorViewMut};

const _: () = assert!(check_constitutive_model_maps::<
    6,
    2,
    7,
    4,
    4,
    MisesPlasticity3D,
>());

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
pub struct MisesPlasticity3D();

create_history_parameter_struct!(
    MisesPlasticityParameters,
    4,
    4,
    [
        (mu, (QDim::Scalar)),
        (kappa, (QDim::Scalar)),
        (y_0, (QDim::Scalar)),
        (h, (QDim::Scalar))
    ]
);
create_history_parameter_struct!(
    MisesPlasticityHistory,
    2,
    7,
    [
        (alpha, (QDim::Scalar)),
        (plastic_strain, (QDim::RotatableVector(6)))
    ]
);

impl ConstitutiveModelFn<6, 2, 7, 4, 4> for MisesPlasticity3D {
    type History = MisesPlasticityHistory;
    type Parameters = MisesPlasticityParameters;

    #[inline]
    fn evaluate(
        _time: f64,
        _del_time: f64,
        //_strain: &[f64; 6],
        del_strain: &[f64; 6],
        stress: &mut [f64; 6],
        tangent: Option<&mut [[f64; 6]; 6]>,
        history: &mut [f64; 7],
        parameters: &[f64; 4],
    ) {
        let mises_parameters = Self::Parameters::from_array(parameters);
        let mu = mises_parameters.mu;
        let kappa = mises_parameters.kappa;
        let y_0 = mises_parameters.y_0;
        let h = mises_parameters.h;

        const SYM_ID: SMatrix<f64, 6, 1> = const { sym_id::<6>() };
        const SYM_ID_OUTER_SYM_ID: SMatrix<f64, 6, 6> = const { sym_id_outer_sym_id::<6>() };
        const PROJECTION_DEV: SMatrix<f64, 6, 6> = const { projection_dev::<6>() };

        // Unpack history
        let history_ = Self::History::from_array_mut(history);
        let alpha = history_.alpha;

        let del_strain_vec = SVectorView::<f64, 6>::from_array(del_strain);
        let mut stress_vec = SVectorViewMut::<f64, 6>::from_array(stress);

        let (p_0, s_0) = stress_vec.vol_dev();
        let (eps_trace, eps_dev) = del_strain_vec.trace_dev();
        let p_1 = p_0 + kappa * eps_trace;

        let s_tr = &s_0 + (2. * mu) * &eps_dev;
        let s_tr_eq = s_tr.mises_norm();

        let sigma_y = y_0 + h * alpha;

        //the .max(0.0) contains the check if the stress is already above the yield surface
        if s_tr_eq < sigma_y {
            // Elastic step
            stress_vec.copy_from(&(p_1 * SYM_ID + s_tr));
            if let Some(tangent) = tangent {
                *tangent = (kappa * &SYM_ID_OUTER_SYM_ID + (2. * mu) * &PROJECTION_DEV)
                    .data
                    .0;
            }
            return;
        } else {
            let del_alpha = (s_tr_eq - sigma_y) / (3. * mu + h);
            let del_gamma = f64::sqrt(3. / 2.) * del_alpha;
            let theta = 1. - (3. * mu * del_alpha) / s_tr_eq;

            // Update the equivalent plastic strain
            // determine the plastic strain
            let n = s_tr / s_tr_eq;
            history_.plastic_strain += del_gamma * &n;
            history_.alpha += del_alpha;

            stress_vec.copy_from(&(p_1 * &SYM_ID + theta * &s_tr));

            if let Some(tangent) = tangent {
                let theta_bar = 1.0 / (1.0 + (h / (3.0 * mu))) - (1.0 - theta);
                let tangent_new = kappa * &SYM_ID_OUTER_SYM_ID
                    + (2.0 * mu * theta) * &PROJECTION_DEV
                    + (2.0 * mu * theta_bar) * &n * &n.transpose();
                // Copy the tangent matrix to the output
                *tangent = tangent_new.data.0;
            }
        }
    }
}
