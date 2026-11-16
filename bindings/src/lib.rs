use comfe::interfaces::*;
use comfe::linear_elasticity::LinearElasticity3D;
use comfe::mises_plasticity::MisesPlasticity3D;
use comfe::plasticity::{DruckerPrager3D, DruckerPragerHyperbolic3D, IsotropicMises3D, IsotropicPlasticityModel3D};
use comfe::finite_strain::{HenckyElasticity3D, HenckyPlasticity3D};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use numpy::{PyReadonlyArray1, PyReadwriteArray1};

use std::collections::HashMap;

#[pyclass]
pub enum StressStrainConstraint {
    UNIAXIAL_STRAIN = 1,
    UNIAXIAL_STRESS = 2,
    PLANE_STRAIN = 3,
    PLANE_STRESS = 4,
    FULL = 5,
}
#[pymethods]
impl StressStrainConstraint {
    #[getter]
    pub const fn stress_strain_dim(&self) -> usize {
        match self {
            StressStrainConstraint::UNIAXIAL_STRAIN => 1,
            StressStrainConstraint::UNIAXIAL_STRESS => 1,
            StressStrainConstraint::PLANE_STRAIN => 4,
            StressStrainConstraint::PLANE_STRESS => 4,
            StressStrainConstraint::FULL => 6,
        }
    }
    #[getter]
    pub const fn geometric_dim(&self) -> usize {
        match self {
            StressStrainConstraint::UNIAXIAL_STRAIN => 1,
            StressStrainConstraint::UNIAXIAL_STRESS => 1,
            StressStrainConstraint::PLANE_STRAIN => 2,
            StressStrainConstraint::PLANE_STRESS => 2,
            StressStrainConstraint::FULL => 3,
        }
    }
}
/// A macro that generates Python bindings for a constitutive model that is somewhat compatible with
/// the interface of (fenics-constitutive)[https://github.com/BAMresearch/fenics-constitutive/]
#[macro_export]
macro_rules! implement_python_model {
    ($m:expr, $name:ident, $model:ty, $constr:expr) => {
        #[pyclass]
        struct $name {
            parameters: <$model as ConstitutiveModelFn<
                { <$model>::STRESS_STRAIN },
                { <$model>::N_HISTORY },
                { <$model>::HISTORY },
                { <$model>::N_PARAMETERS },
                { <$model>::PARAMETERS },
            >>::Parameters,
        }

        #[pymethods]
        impl $name {
            #[new]
            pub fn new(parameters: HashMap<String,PyReadonlyArray1<f64>>) -> Self {
                let mut rust_map = HashMap::new();
                for (key, val) in parameters.iter() {
                    rust_map.insert(key.as_str(),val.as_slice().unwrap());    
                }
                Self {
                    parameters: <$model as ConstitutiveModelFn<
                        { <$model>::STRESS_STRAIN },
                        { <$model>::N_HISTORY },
                        { <$model>::HISTORY },
                        { <$model>::N_PARAMETERS },
                        { <$model>::PARAMETERS },
                    >>::Parameters::from_hashmap(rust_map).unwrap(),
                }
            }
            pub fn evaluate(
                &self,
                time: f64,
                del_time: f64,
                //strain: PyReadonlyArray1<f64>,
                del_grad_u: PyReadonlyArray1<f64>,
                mut stress: PyReadwriteArray1<f64>,
                mut tangent: Option<PyReadwriteArray1<f64>>,
                mut history: Option<HashMap<String, PyReadwriteArray1<f64>>>,
            ) {
                //let strain = strain.as_slice().unwrap();
                let del_grad_u = del_grad_u.as_slice().unwrap();
                let mut stress = stress.as_slice_mut().unwrap();
                let history = history.as_mut();
                let mut history = {
                    let default: &mut [f64] = &mut [];
                    match history {
                        Some(history_map) => history_map
                            .get_mut("history")
                            .expect("'history' entry not found in input")
                            .as_slice_mut()
                            .unwrap(),
                        None => default,
                    }
                };
                //history.as_mut();
                //.get_mut("history")
                //.expect("'history' entry not found in input")
                //.as_slice_mut()
                //.unwrap();
                let parameters = self.parameters.as_array();
                let tangent = tangent.as_mut();
                let tangent = match tangent {
                    Some(tangent) => Some(tangent.as_slice_mut().unwrap()),
                    None => None,
                };
                evaluate_model::<
                    { <$model>::STRESS_STRAIN },
                    { $constr.geometric_dim() },
                    { <$model>::N_HISTORY },
                    { <$model>::HISTORY },
                    { <$model>::N_PARAMETERS },
                    { <$model>::PARAMETERS },
                    $model,
                >(
                    time,
                    del_time,
                    del_grad_u,
                    &mut stress,
                    tangent,
                    &mut history,
                    parameters,
                );
            }
            #[getter]
            pub fn history_dim(&self) -> Option<HashMap<String, usize>> {
                match <$model>::HISTORY {
                    0 => None,
                    _ => Some(HashMap::from([("history".to_string(), <$model>::HISTORY)])),
                }
            }
            #[getter]
            pub fn stress_strain_dim(&self) -> usize {
                self.constraint().stress_strain_dim()
            }
            #[getter]
            pub fn geometric_dim(&self) -> usize {
                self.constraint().geometric_dim()
            }
            #[getter]
            pub fn constraint(&self) -> StressStrainConstraint {
                $constr
            }

            pub fn print_parameters(&self) {
                println!("{:?}", self.parameters)
            }
        }
        $m.add_class::<$name>()?;
    };
}

/// Generates Python bindings for a large-strain model implementing
/// `FiniteStrainConstitutiveModelFn`. Errors of the model are raised as `ValueError`.
#[macro_export]
macro_rules! implement_python_finite_strain_model {
    ($m:expr, $name:ident, $model:ty) => {
        #[pyclass]
        struct $name {
            parameters: <$model as FiniteStrainConstitutiveModelFn<
                { <$model>::N_HISTORY },
                { <$model>::HISTORY },
                { <$model>::N_PARAMETERS },
                { <$model>::PARAMETERS },
            >>::Parameters,
        }

        #[pymethods]
        impl $name {
            #[new]
            pub fn new(parameters: HashMap<String, PyReadonlyArray1<f64>>) -> PyResult<Self> {
                let mut rust_map = HashMap::new();
                for (key, val) in parameters.iter() {
                    rust_map.insert(key.as_str(), val.as_slice()?);
                }
                let parameters = <$model as FiniteStrainConstitutiveModelFn<
                    { <$model>::N_HISTORY },
                    { <$model>::HISTORY },
                    { <$model>::N_PARAMETERS },
                    { <$model>::PARAMETERS },
                >>::Parameters::from_hashmap(rust_map)
                .ok_or_else(|| PyValueError::new_err("missing or malformed parameters"))?;
                Ok(Self { parameters })
            }
            #[pyo3(signature = (time, del_time, f, tau, tangent, history))]
            pub fn evaluate(
                &self,
                time: f64,
                del_time: f64,
                f: PyReadonlyArray1<f64>,
                mut tau: PyReadwriteArray1<f64>,
                mut tangent: Option<PyReadwriteArray1<f64>>,
                mut history: Option<HashMap<String, PyReadwriteArray1<f64>>>,
            ) -> PyResult<()> {
                let f = f.as_slice()?;
                let tau = tau.as_slice_mut()?;
                let tangent = match tangent.as_mut() {
                    Some(tangent) => Some(tangent.as_slice_mut()?),
                    None => None,
                };
                let mut default: [f64; 0] = [];
                let history: &mut [f64] = match history.as_mut() {
                    Some(history_map) => history_map
                        .get_mut("history")
                        .ok_or_else(|| PyValueError::new_err("'history' entry not found in input"))?
                        .as_slice_mut()?,
                    None => &mut default,
                };
                evaluate_finite_strain_model::<
                    { <$model>::N_HISTORY },
                    { <$model>::HISTORY },
                    { <$model>::N_PARAMETERS },
                    { <$model>::PARAMETERS },
                    $model,
                >(
                    time,
                    del_time,
                    f,
                    tau,
                    tangent,
                    history,
                    self.parameters.as_array(),
                )
                .map_err(PyValueError::new_err)
            }
            #[getter]
            pub fn history_dim(&self) -> Option<HashMap<String, usize>> {
                match <$model>::HISTORY {
                    0 => None,
                    _ => Some(HashMap::from([("history".to_string(), <$model>::HISTORY)])),
                }
            }
            /// History of one quadrature point in the undeformed, stress-free state.
            pub fn initial_history(&self) -> Option<HashMap<String, Vec<f64>>> {
                match <$model>::HISTORY {
                    0 => None,
                    _ => Some(HashMap::from([(
                        "history".to_string(),
                        <$model>::initial_history().to_vec(),
                    )])),
                }
            }
            #[getter]
            pub fn geometric_dim(&self) -> usize {
                3
            }

            pub fn print_parameters(&self) {
                println!("{:?}", self.parameters)
            }
        }
        $m.add_class::<$name>()?;
    };
}

#[pymodule]
fn _bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    implement_python_model!(
        m,
        PyLinearElasticity3D,
        LinearElasticity3D,
        StressStrainConstraint::FULL
    );
    implement_python_model!(
        m,
        PyMisesPlasticity3D,
        MisesPlasticity3D,
        StressStrainConstraint::FULL
    );
    implement_python_model!(
        m,
        PyDruckerPrager3D,
        IsotropicPlasticityModel3D<5,5, DruckerPrager3D>,
        StressStrainConstraint::FULL
    );
    implement_python_model!(
        m,
        PyDruckerPragerHyperbolic3D,
        IsotropicPlasticityModel3D<7,7, DruckerPragerHyperbolic3D>,
        StressStrainConstraint::FULL
    );
    implement_python_model!(
        m,
        PyIsotropicMises3D,
        IsotropicPlasticityModel3D<4,4, IsotropicMises3D>,
        StressStrainConstraint::FULL
    );

    implement_python_finite_strain_model!(m, PyHenckyElasticity3D, HenckyElasticity3D);
    implement_python_finite_strain_model!(
        m,
        PyHenckyMises3D,
        HenckyPlasticity3D<4, 4, IsotropicMises3D>
    );
    implement_python_finite_strain_model!(
        m,
        PyHenckyDruckerPrager3D,
        HenckyPlasticity3D<5, 5, DruckerPrager3D>
    );
    implement_python_finite_strain_model!(
        m,
        PyHenckyDruckerPragerHyperbolic3D,
        HenckyPlasticity3D<7, 7, DruckerPragerHyperbolic3D>
    );
    implement_python_finite_strain_model!(
        m,
        PyHenckyMisesCauchyYield3D,
        HenckyPlasticity3D<4, 4, IsotropicMises3D, true>
    );
    implement_python_finite_strain_model!(
        m,
        PyHenckyDruckerPragerCauchyYield3D,
        HenckyPlasticity3D<5, 5, DruckerPrager3D, true>
    );
    implement_python_finite_strain_model!(
        m,
        PyHenckyDruckerPragerHyperbolicCauchyYield3D,
        HenckyPlasticity3D<7, 7, DruckerPragerHyperbolic3D, true>
    );

    Ok(())
}
