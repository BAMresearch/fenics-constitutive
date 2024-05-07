use nalgebra::{
    Const, DVectorView, Dyn, SMatrix, SMatrixView, SMatrixViewMut, SVector, SVectorView,
    SVectorViewMut,
};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

pub fn add_volumetric(mandel: &mut SVector<f64, 6>, p: f64) {
    mandel.x += p;
    mandel.y += p;
    mandel.z += p;
}

pub fn volumetric(mandel: &SVector<f64, 6>) -> f64 {
    //TODO: I don't really like the notation with the x,y,z for the components,
    // but it is probably the fastest
    (mandel.x + mandel.y + mandel.z) / 3.0
}

pub fn mandel_decomposition(mandel: &SVector<f64, 6>) -> (f64, SVector<f64, 6>) {
    let p = volumetric(mandel);
    let mut dev = mandel.clone();
    add_volumetric(&mut dev, p);
    (-p, dev)
}

pub fn strain_from_grad_u(grad_u: &SMatrix<f64, 3, 3>) -> SVector<f64, 6> {
    // creates the strain rate mandel vector directly from the velocity gradient L instead of the rate
    // of deformation tensor D. Therefore, the factor is 1/sqrt(2) instead of sqrt(2)
    const FACTOR: f64 = std::f64::consts::FRAC_1_SQRT_2;

    SVector::<f64, 6>::new(
        grad_u.m11,
        grad_u.m22,
        grad_u.m33,
        FACTOR * (grad_u.m23 + grad_u.m32),
        FACTOR * (grad_u.m13 + grad_u.m31),
        FACTOR * (grad_u.m12 + grad_u.m21),
    )
}

// a type for a function that takes a float and returns a tuple of 2 floats
type HardeningFunction = fn(f64) -> (f64, f64);

struct MisesPlasticity3D {
    mu: f64,
    lambda: f64,
    yield_stress: f64,
    hardening_function: HardeningFunction,
}

impl MisesPlasticity3D {
    fn evaluate_ip(
        &self,
        del_t: f64,
        del_grad_u: &SMatrix<f64, 3, 3>,
        stress: &mut SVector<f64, 6>,
        tangent: &mut SMatrix<f64, 6, 6>,
        history: &mut SVector<f64, 1>,
    ) {
        let del_strain = strain_from_grad_u(del_grad_u);
        let (_, del_strain_dev) = mandel_decomposition(&del_strain);
        let (p, s_0) = mandel_decomposition(&stress);
        let trial_stress = s_0 + 2.0 * self.mu * del_strain_dev;

        let trial_mises = ((3.0 / 2.0) * trial_stress.norm_squared()).sqrt();
        let lambda_0 = history[0];
        let mut del_lambda = 0.0;
        let (mut h, mut dh) = (self.hardening_function)(lambda_0);

        while (trial_mises - 3.0 * self.mu * del_lambda - self.yield_stress - h).abs() > 1e-10 {
            del_lambda += (trial_mises - 3.0 * self.mu * del_lambda - self.yield_stress - h)
                / (-3. * self.mu - dh);
            (h, dh) = (self.hardening_function)(lambda_0 + del_lambda);
        }

        let alpha = 1.0 - 3.0 * self.mu * del_lambda / trial_mises;
        let mut stress_1 = trial_stress * alpha;
        add_volumetric(&mut stress_1, p);
        stress.copy_from(&stress_1);
    }
}
