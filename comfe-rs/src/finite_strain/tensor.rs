//! Spectral functions of symmetric 3x3 matrices that are needed for the Hencky kinematics:
//! eigen decomposition, $\tfrac12\ln$, its derivative and the exponential.
//!
//! `nalgebra::SymmetricEigen` is deliberately not used: for nearly degenerate matrices it
//! reconstructs the input only to about $10^{-9}$, which is visible in the logarithmic strain.
use nalgebra::{Matrix3, Vector3};

/// Eigen decomposition $\mathbf A = \mathbf Q\,\mathrm{diag}(\boldsymbol\lambda)\,\mathbf Q^T$
/// of a symmetric matrix with the cyclic Jacobi method. The eigenvectors are the columns of
/// $\mathbf Q$, the eigenvalues are not sorted. Only the upper triangle of `a` is used.
pub fn sym_eigen(a: &Matrix3<f64>) -> (Vector3<f64>, Matrix3<f64>) {
    const PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];
    let mut a = Matrix3::new(
        a[(0, 0)], a[(0, 1)], a[(0, 2)],
        a[(0, 1)], a[(1, 1)], a[(1, 2)],
        a[(0, 2)], a[(1, 2)], a[(2, 2)],
    );
    let mut q = Matrix3::<f64>::identity();
    for _sweep in 0..50 {
        if PAIRS.iter().all(|&(p, r)| a[(p, r)] == 0.0) {
            break;
        }
        for &(p, r) in PAIRS.iter() {
            let apr = a[(p, r)];
            if apr == 0.0 {
                continue;
            }
            // Off-diagonal entries below round-off of the diagonal are set to zero, which
            // terminates the iteration (Numerical Recipes, `jacobi`).
            if apr.abs() <= 1e-3 * f64::EPSILON * (a[(p, p)].abs() + a[(r, r)].abs()) {
                a[(p, r)] = 0.0;
                a[(r, p)] = 0.0;
                continue;
            }
            let theta = (a[(r, r)] - a[(p, p)]) / (2.0 * apr);
            let t = theta.signum() / (theta.abs() + theta.hypot(1.0));
            let c = 1.0 / t.hypot(1.0);
            let s = t * c;
            let mut rot = Matrix3::<f64>::identity();
            rot[(p, p)] = c;
            rot[(r, r)] = c;
            rot[(p, r)] = s;
            rot[(r, p)] = -s;
            a = rot.transpose() * a * rot;
            a[(p, r)] = 0.0;
            a[(r, p)] = 0.0;
            q *= rot;
        }
    }
    (a.diagonal(), q)
}

/// $\mathbf Q\,\mathrm{diag}(\boldsymbol\lambda)\,\mathbf Q^T$
pub fn from_spectral(lambda: &Vector3<f64>, q: &Matrix3<f64>) -> Matrix3<f64> {
    q * Matrix3::from_diagonal(lambda) * q.transpose()
}

/// $\tfrac12\ln\mathbf b$ of a symmetric positive definite matrix $\mathbf b$.
///
/// The decomposition is computed for $\mathbf X = \mathbf b - \mathbf I$ and the logarithm with
/// `ln_1p`, which is accurate close to the identity. Returns $(\tfrac12\ln\mathbf b, \mathbf x,
/// \mathbf Q)$ where $\mathbf x$ are the eigenvalues of $\mathbf X$, so that the derivative can
/// be computed with [`half_log_divided_differences`]. Returns `None` if $\mathbf b$ is not
/// positive definite.
pub fn half_log_spectral(b: &Matrix3<f64>) -> Option<(Matrix3<f64>, Vector3<f64>, Matrix3<f64>)> {
    let (x, q) = sym_eigen(&(b - Matrix3::identity()));
    if !x.iter().all(|&xi| xi.is_finite() && xi > -1.0) {
        return None;
    }
    let log = x.map(|xi| 0.5 * xi.ln_1p());
    Some((from_spectral(&log, &q), x, q))
}

/// Matrix $\Gamma$ of the first divided differences of $h(\lambda) = \tfrac12\ln\lambda$ at the
/// eigenvalues $\lambda_i = 1 + x_i$ (Daleckii–Krein), $\Gamma_{ij} = (h(\lambda_i) -
/// h(\lambda_j))/(\lambda_i - \lambda_j)$ and $\Gamma_{ii} = h'(\lambda_i)$.
/// Stable for (nearly) equal eigenvalues.
pub fn half_log_divided_differences(x: &Vector3<f64>) -> Matrix3<f64> {
    let mut gamma = Matrix3::<f64>::zeros();
    for i in 0..3 {
        gamma[(i, i)] = 0.5 / (1.0 + x[i]);
        for j in (i + 1)..3 {
            // (ln λ_i - ln λ_j)/(λ_i - λ_j) = ln_1p(r) / (r λ_j) with r = λ_i/λ_j - 1
            let lambda_j = 1.0 + x[j];
            let r = (x[i] - x[j]) / lambda_j;
            let ln_1p_r_over_r = if r.abs() < 1e-8 {
                1.0 - 0.5 * r + r * r / 3.0
            } else {
                r.ln_1p() / r
            };
            let value = 0.5 * ln_1p_r_over_r / lambda_j;
            gamma[(i, j)] = value;
            gamma[(j, i)] = value;
        }
    }
    gamma
}

/// Directional derivative $\mathrm D(\tfrac12\ln)(\mathbf b)[\mathbf H] = \mathbf Q(\Gamma\circ
/// \mathbf Q^T\mathbf H\mathbf Q)\mathbf Q^T$ for a symmetric direction $\mathbf H$, with
/// $\Gamma$ from [`half_log_divided_differences`] and $\mathbf Q$ from [`half_log_spectral`].
pub fn apply_half_log_derivative(
    gamma: &Matrix3<f64>,
    q: &Matrix3<f64>,
    h: &Matrix3<f64>,
) -> Matrix3<f64> {
    let h_rotated = q.transpose() * h * q;
    q * gamma.component_mul(&h_rotated) * q.transpose()
}

/// Exponential of a symmetric matrix.
pub fn expm_sym(a: &Matrix3<f64>) -> Matrix3<f64> {
    let (lambda, q) = sym_eigen(a);
    from_spectral(&lambda.map(f64::exp), &q)
}

/// $\tfrac12(\mathbf A + \mathbf A^T)$
pub fn sym(a: &Matrix3<f64>) -> Matrix3<f64> {
    0.5 * (a + a.transpose())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rotation(axis: Vector3<f64>, angle: f64) -> Matrix3<f64> {
        *nalgebra::Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(axis), angle).matrix()
    }

    fn test_matrices() -> Vec<Matrix3<f64>> {
        let q = rotation(Vector3::new(0.3, -1.0, 0.7), 0.9);
        vec![
            Matrix3::new(2.0, 0.3, -0.1, 0.3, 1.5, 0.4, -0.1, 0.4, 0.8),
            Matrix3::identity(),
            Matrix3::from_diagonal(&Vector3::new(0.5, 2.0, 1.3)),
            // two equal and one nearly equal eigenvalue
            from_spectral(&Vector3::new(1.2, 1.2, 1.2 + 1e-9), &q),
            // nearly the identity
            from_spectral(&Vector3::new(1.0 + 1e-8, 1.0 - 2e-8, 1.0 + 3e-9), &q),
            from_spectral(&Vector3::new(0.3, 4.0, 4.0), &q),
        ]
    }

    #[test]
    fn test_sym_eigen_reconstructs() {
        for a in test_matrices() {
            let (lambda, q) = sym_eigen(&a);
            assert!((from_spectral(&lambda, &q) - a).norm() <= 1e-14 * a.norm());
            assert!((q.transpose() * q - Matrix3::identity()).norm() <= 1e-14);
        }
    }

    #[test]
    fn test_expm_of_log() {
        for b in test_matrices() {
            let (half_log, _, _) = half_log_spectral(&b).unwrap();
            assert!((expm_sym(&(2.0 * half_log)) - b).norm() <= 1e-14 * b.norm());
        }
    }

    #[test]
    fn test_half_log_close_to_identity() {
        let eps = Matrix3::new(1e-9, 2e-10, 0.0, 2e-10, -3e-9, 1e-10, 0.0, 1e-10, 5e-10);
        let b = Matrix3::identity() + 2.0 * eps;
        // b - I is exact, while b itself carries a rounding error of 1e-7 relative to eps
        let x = b - Matrix3::identity();
        let (half_log, _, _) = half_log_spectral(&b).unwrap();
        // 1/2 ln(I + X) = 1/2 (X - X^2/2) + O(X^3)
        let expected = 0.5 * (x - 0.5 * x * x);
        assert!((half_log - expected).norm() <= 1e-14 * x.norm());
    }

    #[test]
    fn test_half_log_not_positive_definite() {
        assert!(half_log_spectral(&Matrix3::zeros()).is_none());
        assert!(half_log_spectral(&Matrix3::from_diagonal(&Vector3::new(1.0, -0.5, 2.0))).is_none());
    }

    #[test]
    fn test_half_log_derivative_finite_differences() {
        let h = Matrix3::new(0.3, -0.2, 0.5, -0.2, 1.0, 0.1, 0.5, 0.1, -0.4);
        let step = 1e-6;
        for b in test_matrices() {
            let (_, x, q) = half_log_spectral(&b).unwrap();
            let gamma = half_log_divided_differences(&x);
            let analytical = apply_half_log_derivative(&gamma, &q, &h);
            let (plus, _, _) = half_log_spectral(&(b + step * h)).unwrap();
            let (minus, _, _) = half_log_spectral(&(b - step * h)).unwrap();
            let fd = (plus - minus) / (2.0 * step);
            assert!(
                (analytical - fd).norm() <= 1e-8 * analytical.norm(),
                "b = {}, analytical = {}, fd = {}",
                b,
                analytical,
                fd
            );
        }
    }
}
