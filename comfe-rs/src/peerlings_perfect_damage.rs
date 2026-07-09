use crate::interfaces::*;
use crate::mandel::*;
use crate::create_history_parameter_struct;
use nalgebra::{SMatrix, SVector, SVectorView, SVectorViewMut};

/// Peerlings gradient damage model with perfect damage behavior for 3D stress states.
///
/// This model is based on the paper "GRADIENT ENHANCED DAMAGE FOR QUASI-BRITTLE MATERIALS"
/// by Peerlings et al. (1996).
///
/// The damage variable $\omega$ is computed from the nonlocal equivalent strain $\bar{\varepsilon}$:
/// - $\omega = 0$ if $\bar{\varepsilon} < \varepsilon_0$
/// - $\omega = 1 - \frac{\varepsilon_0}{\bar{\varepsilon}} \cdot \omega_{max}$ otherwise
///
/// The stress is computed as: $\sigma = (1 - \omega) \cdot C : \varepsilon$
///
/// The local quantity is the Euclidean norm of the total strain tensor.
///
/// # Parameters
/// - `mu`: Shear modulus
/// - `kappa`: Bulk modulus
/// - `eps_0`: Strain threshold at which damage initiates
/// - `omega_max`: Maximum damage value (should be < 1.0 to avoid numerical issues)
#[repr(C)]
pub struct PeerlingsGradientPerfectDamage3D();

create_history_parameter_struct!(
    PeerlingsParameters,
    4,
    4,
    [
        (mu, (QDim::Scalar)),
        (kappa, (QDim::Scalar)),
        (eps_0, (QDim::Scalar)),
        (omega_max, (QDim::Scalar))
    ]
);

create_history_parameter_struct!(
    PeerlingsHistory3D,
    2,
    7,
    [
        (omega, (QDim::Scalar)),
        (total_strain, (QDim::RotatableVector(6)))
    ]
);

impl GradientConstitutiveModelFn<6, 2, 7, 4, 4> for PeerlingsGradientPerfectDamage3D {
    type History = PeerlingsHistory3D;
    type Parameters = PeerlingsParameters;

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
        parameters: &[f64; 4],
    ) {
        // Unpack parameters
        let params = Self::Parameters::from_array(parameters);
        let mu = params.mu;
        let kappa = params.kappa;
        let eps_0 = params.eps_0;
        let omega_max = params.omega_max;

        // Get elastic tangent
        let elastic_tangent: SMatrix<f64, 6, 6> = isotropic_elastic_tangent(mu, kappa);

        // Unpack history
        let history_ = Self::History::from_array_mut(history);

        // Update total strain: total_strain += del_strain
        let del_strain_vec = SVectorView::<f64, 6>::from_array(del_strain);
        history_.total_strain += del_strain_vec;
        let total_strain = history_.total_strain.clone();

        // Compute damage from nonlocal quantity (same law as the Python reference model)
        let eps_eq = nonlocal_quantity[0];
        let omega_new = if eps_eq >= eps_0 {
            (1.0 - eps_0 / eps_eq) * omega_max
        } else {
            0.0
        };

        // Update damage history (damage can only increase)
        let omega_old = history_.omega;
        history_.omega = omega_old.max(omega_new);
        let omega = history_.omega;

        // Compute stress: sigma = (1 - omega) * C * total_strain
        let mut stress_vec = SVectorViewMut::<f64, 6>::from_array(stress);
        stress_vec.copy_from(&((1.0 - omega) * (&elastic_tangent * &total_strain)));

        // Compute local quantity: Euclidean norm of total strain
        local_quantity[0] = total_strain.norm();

        // Compute tangents if requested
        if let Some(tangents) = tangents {
            // dsigma_deps = (1 - omega) * C
            tangents.dsigma_deps = (1.0 - omega) * &elastic_tangent;

            // dlocal_deps = total_strain / ||total_strain|| (gradient of norm)
            let strain_norm = local_quantity[0];
            if strain_norm > 0.0 {
                tangents.dlocal_deps = total_strain.transpose() / strain_norm;
            } else {
                tangents.dlocal_deps = SVector::<f64, 6>::zeros().transpose();
            }

            // dlocal_dnonlocal = 0 (local quantity doesn't depend on nonlocal quantity)
            tangents.dlocal_dnonlocal = 0.0;

            // dsigma_dnonlocal = -domega_dnonlocal * C * total_strain
            // zero while damage is frozen (omega_new below the stored maximum)
            let domega_dnonlocal = if eps_eq >= eps_0 && omega_new >= omega_old {
                (omega_max * eps_0) / (eps_eq * eps_eq)
            } else {
                0.0
            };
            tangents.dsigma_dnonlocal = -domega_dnonlocal * (&elastic_tangent * &total_strain);
        }
    }
}
