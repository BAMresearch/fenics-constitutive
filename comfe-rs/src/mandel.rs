use core::f64;

use crate::consts::*;
use nalgebra::{
    Matrix3, SMatrix, SVector, SVectorView, SVectorViewMut,
};

pub trait Mandel<const DIM: usize> {
    fn trace(&self) -> f64;

    fn vol_dev(&self) -> (f64, SVector<f64, DIM>);

    fn trace_dev(&self) -> (f64, SVector<f64, DIM>);

    fn i_1(&self) -> f64 {
        self.trace()
    }

    fn j_2(&self) -> f64 {
        let (_, dev) = self.trace_dev();
        0.5 * dev.norm_squared()
    }

    fn i1_j2_dev(&self) -> (f64, f64, SVector<f64, DIM>) {
        let (i_1, dev) = self.trace_dev();
        let j_2 = 0.5 * dev.norm_squared();
        (i_1, j_2, dev)
    }

    fn mises_norm(&self) -> f64 {
        let j_2 = self.j_2();
        (3.0 * j_2).sqrt()
    }
}

pub trait MandelMut<const DIM: usize>: Mandel<DIM> {
    fn add_volumetric(&mut self, volumetric: f64);
}

pub trait MandelView<'a, const DIM: usize>: Mandel<DIM> {
    fn from_array(slice: &'a [f64; DIM]) -> Self;
}
pub trait MandelViewMut<'a, const DIM: usize>: MandelMut<DIM> {
    fn from_array(slice: &'a mut [f64; DIM]) -> Self;
}

macro_rules! impl_mandel {
    ($type:ty, $dim:expr) => {
        impl Mandel<$dim> for $type {
            fn trace(&self) -> f64 {
                self.x + self.y + self.z
            }

            fn vol_dev(&self) -> (f64, SVector<f64, $dim>) {
                let trace = self.trace();
                let volumetric = trace / 3.0;
                let mut deviatoric = self.clone_owned();
                deviatoric.add_volumetric(-volumetric);
                (volumetric, deviatoric)
            }

            fn trace_dev(&self) -> (f64, SVector<f64, $dim>) {
                let trace = self.trace();
                let volumetric = trace / 3.0;
                let mut deviatoric = self.clone_owned();
                deviatoric.add_volumetric(-volumetric);
                (trace, deviatoric)
            }
        }
    };
}
macro_rules! impl_mandel_mut {
    ($type:ty, $dim:expr) => {
        impl MandelMut<$dim> for $type {
            fn add_volumetric(&mut self, volumetric: f64) {
                self.x += volumetric;
                self.y += volumetric;
                self.z += volumetric;
            }
        }
    };
}
// Use the macro to implement Mandel for SVector<f64, 6>
impl_mandel_mut!(SVector<f64, 6>, 6);
impl_mandel!(SVector<f64, 6>, 6);
impl_mandel_mut!(SVector<f64, 4>, 4);
impl_mandel!(SVector<f64, 4>, 4);

impl_mandel!(SVectorView<'_, f64, 6>, 6);
impl_mandel!(SVectorView<'_, f64, 4>, 4);

impl_mandel_mut!(SVectorViewMut<'_, f64, 6>, 6);
impl_mandel!(SVectorViewMut<'_, f64, 6>, 6);
impl_mandel_mut!(SVectorViewMut<'_, f64, 4>, 4);
impl_mandel!(SVectorViewMut<'_, f64, 4>, 4);

impl<'a> MandelView<'a, 6> for SVectorView<'a, f64, 6> {
    fn from_array(slice: &'a [f64; 6]) -> Self {
        //safe because the Output is guaranteed to be sized the same as the input
        unsafe { SVectorView::<'a, f64, 6>::from_slice_unchecked(slice, 0) }
    }
}
impl<'a> MandelViewMut<'a, 6> for SVectorViewMut<'a, f64, 6> {
    fn from_array(slice: &'a mut [f64; 6]) -> Self {
        //safe because the Output is guaranteed to be sized the same as the input
        unsafe { SVectorViewMut::<'a, f64, 6>::from_slice_unchecked(slice, 0) }
    }
}
impl<'a> MandelView<'a, 4> for SVectorView<'a, f64, 4> {
    fn from_array(slice: &'a [f64; 4]) -> Self {
        //safe because the Output is guaranteed to be sized the same as the input
        unsafe { SVectorView::<'a, f64, 4>::from_slice_unchecked(slice, 0) }
    }
}
impl<'a> MandelViewMut<'a, 4> for SVectorViewMut<'a, f64, 4> {
    fn from_array(slice: &'a mut [f64; 4]) -> Self {
        //safe because the Output is guaranteed to be sized the same as the input
        unsafe { SVectorViewMut::<'a, f64, 4>::from_slice_unchecked(slice, 0) }
    }
}

/// Determines the elastic tangent matrix in Mandel notation.
/// # Parameters
/// - `mu`: Shear modulus
/// - `kappa`: Bulk modulus
pub fn isotropic_elastic_tangent<const N: usize>(mu: f64, kappa: f64) -> SMatrix<f64, N, N> {
    (2.0 * mu) * const { &projection_dev::<N>() } + (3.0 * kappa) * const { &projection_vol::<N>() }
}

/// Determines the inverse of the elastic tangent matrix in Mandel notation. It does so
/// by calling `isotropic_elastic_tangent` with $\frac{1}{4\mu}$ and $\frac{1}{9\kappa}$
/// to avoid calculating the inverse directly. The creation is therefore basically as fast as
/// the creation of the tangent matrix.
/// # Parameters
/// - `mu`: Shear modulus
/// - `kappa`: Bulk modulus
pub fn isotropic_elastic_tangent_inv<const N: usize>(mu: f64, kappa: f64) -> SMatrix<f64, N, N> {
    let mu_inv = 1.0 / (4.0 * mu);
    let kappa_inv = 1.0 / (9.0 * kappa);
    isotropic_elastic_tangent(mu_inv, kappa_inv)
}

pub fn nonsymmetric_tensor_to_mandel<const GDIM: usize, const SDIM: usize>(
    tensor: [[f64; GDIM]; GDIM],
) -> [f64; SDIM] {
    let mut output = [0.0; SDIM];
    let factor = f64::consts::FRAC_1_SQRT_2;
    const X_INDICES: [usize; 3] = [0, 0, 1];
    const Y_INDICES: [usize; 3] = [1, 2, 2];
    let n_shear: usize = const {
        if GDIM == 3 {
            3
        } else if GDIM == 2 {
            1
        } else if GDIM == 1 {
            0
        } else {
            panic!("Unknowm geometric dimension");
        }
    };
    for i in 0..GDIM {
        // diagonal elements
        output[i] = tensor[i][i];
    }
    for i in 0..n_shear {
        let x_index = X_INDICES[i];
        let y_index = Y_INDICES[i];
        output[3 + i] = factor * (tensor[x_index][y_index] + tensor[y_index][x_index]);
    }
    output
}

/// Converts a symmetric 3x3 matrix to Mandel notation
/// $[a_{11}, a_{22}, a_{33}, \sqrt2 a_{12}, \sqrt2 a_{13}, \sqrt2 a_{23}]$.
/// The off-diagonal entries are symmetrized, as in [`nonsymmetric_tensor_to_mandel`].
pub fn sym_matrix_to_mandel(a: &Matrix3<f64>) -> SVector<f64, 6> {
    // nalgebra stores columns, so `data.0` holds the rows of the transpose; the symmetric
    // part, and hence the Mandel vector, is the same.
    SVector::<f64, 6>::from(nonsymmetric_tensor_to_mandel::<3, 6>(a.data.0))
}

/// Inverse of [`sym_matrix_to_mandel`].
pub fn mandel_to_sym_matrix(v: &SVector<f64, 6>) -> Matrix3<f64> {
    let factor = f64::consts::FRAC_1_SQRT_2;
    let (a12, a13, a23) = (factor * v[3], factor * v[4], factor * v[5]);
    Matrix3::new(v[0], a12, a13, a12, v[1], a23, a13, a23, v[2])
}

#[cfg(test)]
mod tests_mandel {
    use core::f64;

    use nalgebra::ArrayStorage;

    use super::*;

    static MU: f64 = 1.2e+9;
    static KAPPA: f64 = 1.6e+9;
    static LAMBDA: f64 = KAPPA - 2.0 * MU / 3.0;
    static TANGENT_ANALYTICAL: SMatrix<f64, 6, 6> =
        SMatrix::<f64, 6, 6>::from_array_storage(ArrayStorage([
            [2. * MU + LAMBDA, LAMBDA, LAMBDA, 0.0, 0.0, 0.0],
            [LAMBDA, 2. * MU + LAMBDA, LAMBDA, 0.0, 0.0, 0.0],
            [LAMBDA, LAMBDA, 2. * MU + LAMBDA, 0.0, 0.0, 0.0],
            [0., 0., 0., 2. * MU, 0., 0.],
            [0., 0., 0., 0., 2. * MU, 0.],
            [0., 0., 0., 0., 0., 2. * MU],
        ])); //Note that the memory layout is column wise, but the matrix is symmetric, so it does not matter
    static GRADIENT_3D: [[f64; 3]; 3] = [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
    static GRADIENT_2D: [[f64;2];2] = [[1.,2.],[3.,4.]];
    static GRADIENT_1D: [[f64;1];1] = [[1.]];
    static STRAIN_3D: [f64; 6] = [
        1.,
        5.,
        9.,
        (2. + 4.) * f64::consts::FRAC_1_SQRT_2,
        (3. + 7.) * f64::consts::FRAC_1_SQRT_2,
        (6. + 8.) * f64::consts::FRAC_1_SQRT_2,
    ];
    static STRAIN_2D:[f64;4] = [1.,4., 0., (2.+3.)*f64::consts::FRAC_1_SQRT_2];
    static STRAIN_1D:[f64;1] = [1.];

    #[test]
    fn test_tangent() {
        //test that the generator function equals the tangent from wikipedia
        assert!(
            (isotropic_elastic_tangent(MU, KAPPA) - TANGENT_ANALYTICAL).norm()
                < 1e-14 + 1e-14 * TANGENT_ANALYTICAL.norm()
        )
    }
    #[test]
    fn test_elasticity_tangent_inverse() {
        //test that the inverse tangent generator function actually produces an inverse
        let tangent = isotropic_elastic_tangent::<6>(MU, KAPPA);
        let tangent_inv = isotropic_elastic_tangent_inv::<6>(MU, KAPPA);
        assert!((tangent * tangent_inv - SMatrix::<f64, 6, 6>::identity()).norm() < 1e-14)
    }
    #[test]
    fn test_nonsymmetric_to_mandel_3d() {
        let strain: [f64; 6] = nonsymmetric_tensor_to_mandel(GRADIENT_3D);
        let strain_vec = SVector::<f64, 6>::from_column_slice(&strain);
        let strain_solution = SVector::<f64, 6>::from_column_slice(&STRAIN_3D);
        assert!((strain_vec - strain_solution).norm() < 1e-14);
    }
    #[test]
    fn test_nonsymmetric_to_mandel_2d() {
        let strain: [f64; 4] = nonsymmetric_tensor_to_mandel(GRADIENT_2D);
        let strain_vec = SVector::<f64, 4>::from_column_slice(&strain);
        let strain_solution = SVector::<f64, 4>::from_column_slice(&STRAIN_2D);
        assert!((strain_vec - strain_solution).norm() < 1e-14);
    }
    #[test]
    fn test_sym_matrix_mandel_round_trip() {
        let a = Matrix3::new(1., 2., 3., 2., 5., 6., 3., 6., 9.);
        let v = sym_matrix_to_mandel(&a);
        let sq2 = f64::consts::SQRT_2;
        let expected = SVector::<f64, 6>::from_column_slice(&[1., 5., 9., 2. * sq2, 3. * sq2, 6. * sq2]);
        assert!((v - expected).norm() < 1e-14);
        assert!((mandel_to_sym_matrix(&v) - a).norm() < 1e-14);
        // consistent with the conversion of displacement gradients
        let from_grad: [f64; 6] = nonsymmetric_tensor_to_mandel([[1., 2., 3.], [2., 5., 6.], [3., 6., 9.]]);
        assert!((v - SVector::<f64, 6>::from_column_slice(&from_grad)).norm() < 1e-14);
    }
    #[test]
    fn test_nonsymmetric_to_mandel_1d() {
        let strain: [f64; 1] = nonsymmetric_tensor_to_mandel(GRADIENT_1D);
        let strain_vec = SVector::<f64, 1>::from_column_slice(&strain);
        let strain_solution = SVector::<f64, 1>::from_column_slice(&STRAIN_1D);
        assert!((strain_vec - strain_solution).norm() < 1e-14);
    }

}
