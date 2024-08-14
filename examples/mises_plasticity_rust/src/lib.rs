use nalgebra::{
    Const, DVectorView, Dyn, Matrix, SMatrix, SMatrixView, SMatrixViewMut, SVector, SVectorView,
    SVectorViewMut, Storage,
};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

const SYM_ID: SVector<f64,6> = SVector::<f64, 6>::new(1.0, 1.0, 1.0, 0.0, 0.0, 0.0);
const ID_6: SMatrix<f64, 6, 6> = SMatrix::<f64, 6, 6>::new(
    1., 0., 0., 0., 0., 0.,
    0., 1., 0., 0., 0., 0.,
    0., 0., 1., 0., 0., 0.,
    0., 0., 0., 1., 0., 0.,
    0., 0., 0., 0., 1., 0.,
    0., 0., 0., 0., 0., 1.
);
const SYM_ID_OUTER_SYM_ID: SMatrix<f64, 6,6> = SMatrix::<f64, 6,6>::new(
    1., 1., 1., 0., 0., 0.,
    1., 1., 1., 0., 0., 0.,
    1., 1., 1., 0., 0., 0.,
    0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0., 0.,
);
const PROJECTION_DEV: SMatrix<f64, 6, 6> = SMatrix::<f64, 6,6>::new(
    2./3., -1./3., -1./3., 0., 0., 0.,
    -1./3., 2./3., -1./3., 0., 0., 0.,
    -1./3., -1./3., 2./3., 0., 0., 0.,
    0., 0., 0., 1., 0., 0.,
    0., 0., 0., 0., 1., 0.,
    0., 0., 0., 0., 0., 1.,
);

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

pub fn strain_from_grad_u(grad_u: &SMatrixView<f64, 3, 3>) -> SVector<f64, 6> {
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
type ConstitutiveModel<
    const STRESS_STRAIN: usize,
    const TANGENT: usize,
    const HISTORY: usize,
    const PARAMETERS: usize,
> = fn(
    f64,
    f64,
    &[f64; 9],
    &mut [f64; STRESS_STRAIN],
    &mut [f64; TANGENT],
    &mut [f64; HISTORY],
    &[f64; PARAMETERS],
);

struct MisesPlasticity3D {
    mu: f64,
    lambda: f64,
    yield_stress: f64,
    hardening_function: HardeningFunction,
}

impl MisesPlasticity3D {
    fn evaluate_ip(
        &self,
        t: f64,
        del_t: f64,
        del_grad_u: &[f64; 9],
        stress: &mut [f64; 6],
        tangent: &mut [f64; 36],
        history: &mut [f64; 1],
    ) {
        //convert the fixed-length slices to Matrices and Vectors without copying
        //TODO: I don't like the unsafe blocks, but I don't know how to do it better
        let del_grad_u_view =
            unsafe { SMatrixView::<f64, 3, 3>::from_slice_unchecked(del_grad_u, 0) };
        let stress_view = unsafe { SVectorViewMut::<f64, 6>::from_slice_unchecked(stress, 0) };
        let tangent_view = unsafe { SMatrixViewMut::<f64, 6, 6>::from_slice_unchecked(tangent, 0) };
        let history_view = unsafe { SVectorViewMut::<f64, 1>::from_slice_unchecked(history, 0) };

        let del_strain = strain_from_grad_u(&del_grad_u_view);
        let (_, del_strain_dev) = mandel_decomposition(&del_strain);
        let (p, s_0) = mandel_decomposition(&stress_view.into_owned());

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

        stress[0..6].copy_from_slice(&stress_1.as_slice());
        history[0] = lambda_0 + del_lambda;
    }
}
