use crate::interfaces::*;
use crate::mandel::*;
use crate::{create_history_parameter_struct};
use core::ffi::c_double;
use nalgebra::{SMatrix, SVectorView, SVectorViewMut};

/// Parameter structs of models with isotropic linear elasticity, defined by the shear modulus
/// $\mu$ and the bulk modulus $\kappa$. For parameter structs with fields named `mu` and
/// `kappa`, use [`impl_isotropic_elastic_parameters!`](crate::impl_isotropic_elastic_parameters).
pub trait IsotropicElasticParameters {
    fn mu(&self) -> f64;
    fn kappa(&self) -> f64;

    /// Elastic tangent in Mandel notation.
    fn elastic_tangent<const N: usize>(&self) -> SMatrix<f64, N, N> {
        isotropic_elastic_tangent(self.mu(), self.kappa())
    }
}

/// Implements [`IsotropicElasticParameters`] for a parameter struct with the fields `mu` and
/// `kappa`.
#[macro_export]
macro_rules! impl_isotropic_elastic_parameters {
    ($type:ty) => {
        impl $crate::linear_elasticity::IsotropicElasticParameters for $type {
            #[inline]
            fn mu(&self) -> f64 {
                self.mu
            }
            #[inline]
            fn kappa(&self) -> f64 {
                self.kappa
            }
        }
    };
}

const _: () = assert!(check_constitutive_model_maps::<
    6,
    0,
    0,
    2,
    2,
    LinearElasticity3D,
>());

/// A linear elasticity model for 3D stress states.
///
/// This struct implements Hooke's law for isotropic linear elastic materials.
/// The stress-strain relationship is given by: $\sigma = C : \varepsilon$, where:
/// - $\sigma$ is the stress tensor
/// - $C$ is the fourth-order elastic stiffness tensor
/// - $\varepsilon$ is the strain tensor
///
/// The elastic stiffness tensor is defined in terms of the shear modulus $\mu$ 
/// and bulk modulus $\kappa$.
///
/// # Parameters
/// - `mu`: Shear modulus
/// - `kappa`: Bulk modulus
#[repr(C)]
pub struct LinearElasticity3D();

create_history_parameter_struct!(
    LinearElasticityParameters,
    2,
    2,
    [(mu, (QDim::Scalar)), (kappa, (QDim::Scalar))]
);
crate::impl_isotropic_elastic_parameters!(LinearElasticityParameters);

create_history_parameter_struct!(LinearElasticityHistory, 0, 0, []);

impl ConstitutiveModelFn<6, 0, 0, 2, 2> for LinearElasticity3D {
    //const PARAMETERS_MAP: [(&'static str, QDim); 2] = LinearElasticityParameters::MAP;
    //const HISTORY_MAP: [(&'static str, QDim); 0] = LinearElasticityHistory::MAP;

    type History = LinearElasticityHistory;
    type Parameters = LinearElasticityParameters;
    #[inline]
    fn evaluate(
        _time: f64,
        _del_time: f64,
        //_strain: &[f64; 6],
        del_strain: &[f64; 6],
        stress: &mut [f64; 6],
        tangent: Option<&mut [[f64; 6]; 6]>,
        _history: &mut [f64; 0],
        parameters: &[f64; 2],
    ) {
        // Unpack parameters
        let parameters_ = LinearElasticityParameters::from_array(parameters);
        let elastic_tangent = parameters_.elastic_tangent::<6>();

        let del_strain_vec = SVectorView::<f64, 6>::from_array(del_strain);
        let mut stress_vec = SVectorViewMut::<f64, 6>::from_array(stress);

        stress_vec += &elastic_tangent * &del_strain_vec;

        if let Some(tangent) = tangent {
            *tangent = elastic_tangent.data.0;
        }
    }
}

pub unsafe fn linear_elasticity3d_fn(
    time: c_double,
    del_time: c_double,
    //strain: *const c_double,
    del_strain: *const c_double,
    stress: *mut c_double,
    tangent: *const c_double,
    history: *mut c_double,
    parameters: *const c_double,
) {
    //let strain = unsafe { &*(strain as *const [f64; 6]) };
    let del_strain = unsafe { &*(del_strain as *const [f64; 6]) };
    let stress = unsafe { &mut *(stress as *mut [f64; 6]) };
    let tangent = Some(unsafe { &mut *(tangent as *mut [[f64; 6]; 6]) });
    let history = unsafe { &mut *(history as *mut [f64; 0]) };
    let parameters = unsafe { &*(parameters as *const [f64; 2]) };
    LinearElasticity3D::evaluate(
        time, del_time, del_strain, stress, tangent, history, parameters,
    );
}

#[cfg(test)]
mod tests_mandel {
    use std::collections::HashMap;

    use super::*;
    
    #[test]
    fn test_parameter_from_hashmap() {
        let mut p = HashMap::new();
        p.insert("mu", [42.].as_slice());
        p.insert("kappa", [17.].as_slice());
        let param = LinearElasticityParameters::from_hashmap(p).unwrap();
        assert!(param.mu == 42. && param.kappa == 17.);

    }

}
