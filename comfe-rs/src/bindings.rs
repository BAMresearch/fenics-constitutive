// use crate::plasticity::IsotropicPlasticityModel3D;
// //#[cfg(feature = "python-bindings")]
// use crate::interfaces::{ArrayEquivalent, ConstitutiveModelFn};
// use crate::linear_elasticity::LinearElasticity3D;
// use crate::mises_plasticity::MisesPlasticity3D;
// use crate::plasticity::DruckerPrager3D;
// #[cfg(feature = "python-bindings")]
// use numpy::{PyReadonlyArray1, PyReadwriteArray1};
// #[cfg(feature = "python-bindings")]
// use pyo3::{pyclass, pymethods};
// use std::collections::HashMap;

// #[cfg(feature = "python-bindings")]
// #[pyclass]
// pub enum StressStrainConstraint {
//     UNIAXIAL_STRAIN = 1,
//     UNIAXIAL_STRESS = 2,
//     PLANE_STRAIN = 3,
//     PLANE_STRESS = 4,
//     FULL = 5,
// }
// #[cfg(feature = "python-bindings")]
// #[pymethods]
// impl StressStrainConstraint {
//     #[getter]
//     pub fn stress_strain_dim(&self) -> usize {
//         match self {
//             StressStrainConstraint::UNIAXIAL_STRAIN => 1,
//             StressStrainConstraint::UNIAXIAL_STRESS => 1,
//             StressStrainConstraint::PLANE_STRAIN => 4,
//             StressStrainConstraint::PLANE_STRESS => 4,
//             StressStrainConstraint::FULL => 6,
//         }
//     }
//     #[getter]
//     pub fn geometric_dim(&self) -> usize {
//         match self {
//             StressStrainConstraint::UNIAXIAL_STRAIN => 1,
//             StressStrainConstraint::UNIAXIAL_STRESS => 1,
//             StressStrainConstraint::PLANE_STRAIN => 2,
//             StressStrainConstraint::PLANE_STRESS => 2,
//             StressStrainConstraint::FULL => 3,
//         }
//     }
// }

// /// A macro that generates Python bindings for a constitutive model that is somewhat compatible with
// /// the interface of (fenics-constitutive)[https://github.com/BAMresearch/fenics-constitutive/]
// #[cfg(feature = "python-bindings")]
// #[macro_export]
// macro_rules! implement_python_model {
//     ($name:ident, $model:ty, $constr:expr) => {
//         #[pyclass]
//         struct $name {
//             parameters: <$model as ConstitutiveModelFn<
//                 { <$model>::STRESS_STRAIN },
//                 { <$model>::N_HISTORY },
//                 { <$model>::HISTORY },
//                 { <$model>::N_PARAMETERS },
//                 { <$model>::PARAMETERS },
//             >>::Parameters,
//         }

//         #[pymethods]
//         impl $name {
//             #[new]
//             pub fn new(parameters: PyReadonlyArray1<f64>) -> Self {
//                 let parameter_slice: &[f64; <$model>::PARAMETERS] =
//                     parameters.as_slice().unwrap().try_into().unwrap();
//                 Self {
//                     parameters: *<$model as ConstitutiveModelFn<
//                         { <$model>::STRESS_STRAIN },
//                         { <$model>::N_HISTORY },
//                         { <$model>::HISTORY },
//                         { <$model>::N_PARAMETERS },
//                         { <$model>::PARAMETERS },
//                     >>::Parameters::from_array(parameter_slice),
//                 }
//             }
//             pub fn evaluate(
//                 &self,
//                 time: f64,
//                 del_time: f64,
//                 strain: PyReadonlyArray1<f64>,
//                 del_strain: PyReadonlyArray1<f64>,
//                 mut stress: PyReadwriteArray1<f64>,
//                 mut tangent: Option<PyReadwriteArray1<f64>>,
//                 mut history: PyReadwriteArray1<f64>,
//                 //parameters: PyReadonlyArray1<f64>,
//             ) {
//                 let strain = strain.as_slice().unwrap();
//                 let del_strain = del_strain.as_slice().unwrap();
//                 let mut stress = stress.as_slice_mut().unwrap();
//                 let mut history = history.as_slice_mut().unwrap();
//                 let parameters = self.parameters.as_array();
//                 let tangent = tangent.as_mut();
//                 let tangent = match tangent {
//                     Some(tangent) => Some(tangent.as_slice_mut().unwrap()),
//                     None => None,
//                 };
//                 <$model>::evaluate_all(
//                     time,
//                     del_time,
//                     strain,
//                     del_strain,
//                     &mut stress,
//                     tangent,
//                     &mut history,
//                     parameters,
//                 );
//             }
//             #[getter]
//             pub fn history_dim(&self) -> HashMap<&str, usize> {
//                 HashMap::from([("history", <$model>::HISTORY)])
//             }
//             #[getter]
//             pub fn stress_strain_dim(&self) -> usize {
//                 self.constraint().stress_strain_dim()
//             }
//             #[getter]
//             pub fn geometric_dim(&self) -> usize {
//                 self.constraint().geometric_dim()
//             }
//             #[getter]
//             pub fn constraint(&self) -> StressStrainConstraint {
//                 $constr
//             }
//         }
//     };
// }

// mod tests_bindings {
//     use super::*;

//     #[test]
//     fn test_python() {
//         #[cfg(feature = "python-bindings")]
//         implement_python_model!(
//             PyMisesPlasticity3D,
//             MisesPlasticity3D,
//             StressStrainConstraint::FULL
//         );
//         #[cfg(feature = "python-bindings")]
//         implement_python_model!(
//             PyLinearElasticity3D,
//             LinearElasticity3D,
//             StressStrainConstraint::FULL
//         );
//         #[cfg(feature = "python-bindings")]
//         implement_python_model!(PyDruckerPrager3D, IsotropicPlasticityModel3D<5,5,DruckerPrager3D>, StressStrainConstraint::FULL);
//     }
// }
