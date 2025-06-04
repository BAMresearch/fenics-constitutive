use std::collections::HashMap;

use nalgebra::{Const, DVectorView, Dyn, SMatrix, SVector, SVectorView};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

pub fn strain_from_grad_u(grad_u: &SMatrix<f64, 3, 3>) -> SVector<f64, 6> {
    // creates the strain rate mandel vector directly from the velocity gradient L instead of the rate
    // of deformation tensor D. Therefore, the factor is 1/sqrt(2) instead of sqrt(2)
    const FACTOR: f64 = std::f64::consts::FRAC_1_SQRT_2;

    SVector::<f64, 6>::new(
        grad_u.m11,
        grad_u.m22,
        grad_u.m33,
        FACTOR * (grad_u.m12 + grad_u.m21),
        FACTOR * (grad_u.m13 + grad_u.m31),
        FACTOR * (grad_u.m23 + grad_u.m32),
    )
}

#[pyclass]
struct Elasticity3D {
    D: SMatrix<f64, 6, 6>,
}
#[pymethods]
impl Elasticity3D {
    #[new]
    fn new(E: f64, nu: f64) -> PyResult<Self> {
        let mut D = SMatrix::<f64, 6, 6>::zeros();
        let lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let mu = E / (2.0 * (1.0 + nu));
        D[(0, 0)] = lambda + 2.0 * mu;
        D[(0, 1)] = lambda;
        D[(0, 2)] = lambda;
        D[(1, 0)] = lambda;
        D[(1, 1)] = lambda + 2.0 * mu;
        D[(1, 2)] = lambda;
        D[(2, 0)] = lambda;
        D[(2, 1)] = lambda;
        D[(2, 2)] = lambda + 2.0 * mu;
        D[(3, 3)] = 2.0 * mu;
        D[(4, 4)] = 2.0 * mu;
        D[(5, 5)] = 2.0 * mu;
        Ok(Self { D })
    }
    fn evaluate(
        &self,
        t: f64,
        del_t: f64,
        del_grad_u: PyReadonlyArray1<f64>,
        stress: PyReadwriteArray1<f64>,
        tangent: PyReadwriteArray1<f64>,
        history: HashMap<String, PyReadwriteArray1<f64>>,
    ) -> PyResult<()> {
        //convert numpy arrays to nalgebra matrices
        let del_grad_u = del_grad_u
            .try_as_matrix::<Dyn, Const<1>, Const<1>, Dyn>()
            .unwrap();
        let mut stress = stress
            .try_as_matrix_mut::<Dyn, Const<1>, Const<1>, Dyn>()
            .unwrap();
        let mut tangent = tangent
            .try_as_matrix_mut::<Dyn, Const<1>, Const<1>, Dyn>()
            .unwrap();
        let n_ip = del_grad_u.nrows() / 9;

        for ip in 0..n_ip {
            let mut view_stress = stress.fixed_view_mut::<6, 1>(ip * 6, 0);
            let mut view_tangent = tangent.fixed_view_mut::<36, 1>(ip * 36, 0);
            
            //create matrix from column slice. This is actually false because fenics is row-wise and
            // nalgebra column-wise, but it makes no difference for the strain calculation
            let del_grad_u_full = SMatrix::<f64, 3, 3>::from_column_slice(
                del_grad_u.fixed_view::<9, 1>(ip * 9, 0).as_slice(),
            );

            view_stress += self.D * strain_from_grad_u(&del_grad_u_full);
            view_tangent.copy_from_slice(&self.D.as_slice());
        }
        Ok(())
    }
    fn history_dim(&self) -> HashMap<String, i32> {
        HashMap::new()
    }
}

#[pymodule]
fn elasticity_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Elasticity3D>()?;
    Ok(())
}
