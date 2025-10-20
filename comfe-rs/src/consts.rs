use nalgebra::{ArrayStorage, SMatrix, SVector};


/// Returns the identity matrix in Mandel notation.
/// N represents the size of the Mandel notation.
pub const fn sym_id<const N: usize>() -> SVector<f64, N> {
    let mut data = [0.0; N];
    data[0] = 1.0;
    if N > 1 {
        data[1] = 1.0;
    }
    if N > 2 {
        data[2] = 1.0;
    }
    let array_storage = ArrayStorage::<f64, N, 1>([data]);
    let output = SVector::<f64, N>::from_array_storage(array_storage);
    output
}

/// Returns the outer product of the Mandel notation identity with itself.
/// N represents the size of the Mandel notation. This is equivalent to
/// the outer product of the two second order tensors.
pub const fn sym_id_outer_sym_id<const N: usize>() -> SMatrix<f64, N, N> {
    let mut data = [[0.0; N]; N];
    data[0][0] = 1.0;
    if N > 1 {
        data[0][1] = 1.0;
        data[1][0] = 1.0;
        data[1][1] = 1.0;
    }
    if N > 2 {
        data[0][2] = 1.0;
        data[2][0] = 1.0;
        data[1][2] = 1.0;
        data[2][1] = 1.0;
        data[2][2] = 1.0;
    }
    let storage = ArrayStorage::<f64, N, N>(data);
    SMatrix::<f64, N, N>::from_array_storage(storage)
}

/// Returns the identity matrix of size N x N.
pub const fn id<const N: usize>() -> SMatrix<f64, N, N> {
    let mut data = [[0.0; N]; N];
    let mut i = 0;
    while i < N {
        data[i][i] = 1.0;
        i += 1;
    }
    let storage = ArrayStorage::<f64, N, N>(data);
    SMatrix::<f64, N, N>::from_array_storage(storage)
}

/// Adds two matrices of size N x M.
/// N is the number of rows and M is the number of columns.
/// This function only exists to add matrices in const contexts,
/// as the standard `+` operator is a trait implementation which cannot be used in const contexts.
pub const fn const_add_matrices<const N: usize, const M: usize>(
    a: SMatrix<f64, N, M>,
    b: SMatrix<f64, N, M>,
) -> SMatrix<f64, N, M> {
    let mut c_data = [[0.0; N]; M];
    let a_data = a.data.0;
    let b_data = b.data.0;
    let mut i = 0;
    while i < N {
        let mut j = 0;
        while j < M {
            c_data[i][j] = a_data[i][j] + b_data[i][j];
            j += 1;
        }
        i += 1;
    }
    let storage = ArrayStorage::<f64, N, M>(c_data);
    SMatrix::<f64, N, M>::from_array_storage(storage)
}

/// Multiplies a matrix of size N x M by a scalar.
/// N is the number of rows and M is the number of columns.
/// This function only exists to multiply matrices in const contexts,
/// as the standard `*` operator is a trait implementation which cannot be used in const contexts.
pub const fn const_scalar_mult<const N: usize, const M: usize>(
    a: SMatrix<f64, N, M>,
    scalar: f64,
) -> SMatrix<f64, N, M> {
    let mut c_data = [[0.0; N]; M];
    let a_data = a.data.0;
    let mut i = 0;
    while i < N {
        let mut j = 0;
        while j < M {
            c_data[i][j] = a_data[i][j] * scalar;
            j += 1;
        }
        i += 1;
    }
    let storage = ArrayStorage::<f64, N, M>(c_data);
    SMatrix::<f64, N, M>::from_array_storage(storage)
}

/// Returns the projection matrix onto the volumetric space in Mandel notation.
/// This is the projection to the space spanned by the identity matrix in Mandel notation.
/// N represents the size of the Mandel notation.
/// This is the same matrix as the outer product of the identity matrix with itself, but scaled by 1/3
/// to be a projection matrix.
pub const fn projection_vol<const N:usize>() ->SMatrix<f64,N,N> {
    const_scalar_mult(sym_id_outer_sym_id::<N>(), 1.0 / 3.0)
}

/// Returns the projection matrix onto the deviatoric space in Mandel notation.
/// This is the projection to the space orthogonal to the volumetric space.
/// N represents the size of the Mandel notation.
pub const fn projection_dev<const N: usize>() -> SMatrix<f64, N, N> {
    const_add_matrices(id::<N>(),const_scalar_mult(projection_vol(),-1.0))
}

mod tests_consts {
    use super::*;


    #[test]
    fn test_projections() {
        // test that the projection to the deviatoric space and the volumetric space are orthogonal
        let sym_outer_sym = sym_id_outer_sym_id::<6>();
        let proj_dev = projection_dev::<6>();
        let proj_vol = projection_vol::<6>();
        assert!((sym_outer_sym * proj_dev).norm() < 1e-14);
        assert!((proj_vol * proj_dev).norm() < 1e-14);
        assert!((proj_vol * proj_vol - proj_vol).norm() < 1e-14);
        assert!((proj_dev * proj_dev - proj_dev).norm() < 1e-14);
    }
}
