//! Spectral functions of symmetric 3x3 matrices that are needed for the Hencky kinematics:
//! eigen decomposition, $\tfrac12\ln$, its derivative and the exponential.
//!
//! `nalgebra::SymmetricEigen` is deliberately not used: for nearly degenerate matrices it
//! reconstructs the input only to about $10^{-9}$, which is visible in the logarithmic strain.
use nalgebra::{Matrix3, SMatrix, Vector3};

/// Eigen decomposition $\mathbf A = \mathbf Q\thinspace \mathrm{diag}(\boldsymbol\lambda)\thinspace \mathbf Q^T$
/// of a symmetric matrix with the cyclic Jacobi method. The eigenvectors are the columns of
/// $\mathbf Q$, the eigenvalues are not sorted. Only the upper triangle of `a` is used.
///
/// The rotations are applied in place (Numerical Recipes, `jacobi`): a rotation in the plane
/// $(p, r)$ only changes $a_{pp}$, $a_{rr}$, the two off-diagonal entries coupling $p$ and $r$ to the
/// third index, and two columns of $\mathbf Q$.
pub fn sym_eigen(a: &Matrix3<f64>) -> (Vector3<f64>, Matrix3<f64>) {
    const PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];
    // off-diagonal entry (i, j), i != j, is stored at index i + j - 1
    let mut diagonal = [a[(0, 0)], a[(1, 1)], a[(2, 2)]];
    let mut off_diagonal = [a[(0, 1)], a[(0, 2)], a[(1, 2)]];
    let mut q = Matrix3::<f64>::identity();
    for _sweep in 0..50 {
        if off_diagonal.iter().all(|&value| value == 0.0) {
            break;
        }
        for &(p, r) in PAIRS.iter() {
            let pr = p + r - 1;
            let apr = off_diagonal[pr];
            if apr == 0.0 {
                continue;
            }
            // Off-diagonal entries below round-off of the diagonal are set to zero, which
            // terminates the iteration.
            if apr.abs() <= 1e-3 * f64::EPSILON * (diagonal[p].abs() + diagonal[r].abs()) {
                off_diagonal[pr] = 0.0;
                continue;
            }
            let theta = (diagonal[r] - diagonal[p]) / (2.0 * apr);
            let t = if theta.abs() > 1e150 {
                0.5 / theta
            } else {
                theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
            };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            let tau = s / (1.0 + c);
            diagonal[p] -= t * apr;
            diagonal[r] += t * apr;
            off_diagonal[pr] = 0.0;
            let k = 3 - p - r;
            let (kp, kr) = (k + p - 1, k + r - 1);
            let (g, h) = (off_diagonal[kp], off_diagonal[kr]);
            off_diagonal[kp] = g - s * (h + g * tau);
            off_diagonal[kr] = h + s * (g - h * tau);
            for row in 0..3 {
                let (g, h) = (q[(row, p)], q[(row, r)]);
                q[(row, p)] = g - s * (h + g * tau);
                q[(row, r)] = h + s * (g - h * tau);
            }
        }
    }
    (Vector3::from(diagonal), q)
}

/// $\mathbf Q\thinspace \mathrm{diag}(\boldsymbol\lambda)\thinspace \mathbf Q^T$
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
/// h(\lambda_j))/(\lambda_i - \lambda_j)$ and $\Gamma_{ii} = h^\prime(\lambda_i)$.
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

/// Mandel representation $\mathbf R_6$ of the rotation $\mathbf A \mapsto \mathbf Q\mathbf A\mathbf Q^T$ of symmetric
/// matrices: `mandel(Q A Q^T) = R6 * mandel(A)`. Column $b$ is the Mandel vector of
/// $\mathbf Q\mathbf E_b\mathbf Q^T$ for the orthonormal Mandel basis tensors $\mathbf E_b$, so $\mathbf R_6$ is
/// orthogonal if $\mathbf Q$ is.
pub fn mandel_rotation(q: &Matrix3<f64>) -> SMatrix<f64, 6, 6> {
    const SHEAR: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];
    let sqrt2 = std::f64::consts::SQRT_2;
    let mut r6 = SMatrix::<f64, 6, 6>::zeros();
    // Mandel vector of the symmetric tensor u ⊗ v + v ⊗ u, scaled by `scale`
    let mut set_column = |column: usize, u: usize, v: usize, scale: f64| {
        for m in 0..3 {
            r6[(m, column)] = scale * 2.0 * q[(m, u)] * q[(m, v)];
        }
        for (index, &(m, n)) in SHEAR.iter().enumerate() {
            r6[(3 + index, column)] =
                scale * sqrt2 * (q[(m, u)] * q[(n, v)] + q[(m, v)] * q[(n, u)]);
        }
    };
    for k in 0..3 {
        // E_k = e_k ⊗ e_k
        set_column(k, k, k, 0.5);
    }
    for (index, &(i, j)) in SHEAR.iter().enumerate() {
        // E = (e_i ⊗ e_j + e_j ⊗ e_i) / sqrt 2
        set_column(3 + index, i, j, std::f64::consts::FRAC_1_SQRT_2);
    }
    r6
}

/// Exponential of a symmetric matrix.
pub fn expm_sym(a: &Matrix3<f64>) -> Matrix3<f64> {
    let (lambda, q) = sym_eigen(a);
    from_spectral(&lambda.map(f64::exp), &q)
}

/// Radius (Frobenius norm of $\mathbf b - \mathbf I$) within which [`half_log_pade`] is accurate to
/// round-off: the relative error of the [4/4] approximant of $\ln(1+x)$ is below $1.1\cdot10^{-15}$
/// for $|x|\le 0.05$.
pub const PADE_LOG_RADIUS: f64 = 0.05;

/// Radius (Frobenius norm of $\mathbf A$) within which [`expm_sym_pade`] is accurate to round-off:
/// the relative error of the [4/4] approximant of $e^x$ is below $4\cdot10^{-17}$ for $|x|\le 0.1$.
pub const PADE_EXP_RADIUS: f64 = 0.1;

/// $\tfrac12\ln\mathbf b$ with the diagonal [4/4] Padé approximant
/// $\ln(\mathbf I+\mathbf X)\approx\mathbf Q(\mathbf X)^{-1}\mathbf P(\mathbf X)$, $\mathbf X = \mathbf b - \mathbf I$, which
/// needs no eigen decomposition. Returns `None` if $\lVert\mathbf X\rVert > $ [`PADE_LOG_RADIUS`].
/// `b` is symmetrized first.
pub fn half_log_pade(b: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    let id = Matrix3::<f64>::identity();
    let x = sym(b) - id;
    if !(x.norm() <= PADE_LOG_RADIUS) {
        return None;
    }
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x2 * x2;
    let p = x + 1.5 * x2 + (13.0 / 21.0) * x3 + (5.0 / 84.0) * x4;
    let q = id + 2.0 * x + (9.0 / 7.0) * x2 + (2.0 / 7.0) * x3 + (1.0 / 70.0) * x4;
    // P and Q are polynomials in X and commute, so Q^-1 P is symmetric up to round-off
    Some(0.5 * sym(&(q.try_inverse()? * p)))
}

/// Exponential of a symmetric matrix with the diagonal [4/4] Padé approximant
/// $\mathbf N(-\mathbf A)^{-1}\mathbf N(\mathbf A)$. Returns `None` if
/// $\lVert\mathbf A\rVert > $ [`PADE_EXP_RADIUS`].
pub fn expm_sym_pade(a: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    if !(a.norm() <= PADE_EXP_RADIUS) {
        return None;
    }
    let id = Matrix3::<f64>::identity();
    let a2 = a * a;
    let a3 = a2 * a;
    let a4 = a2 * a2;
    let even = id + (3.0 / 28.0) * a2 + (1.0 / 1680.0) * a4;
    let odd = 0.5 * a + (1.0 / 84.0) * a3;
    Some(sym(&((even - odd).try_inverse()? * (even + odd))))
}

/// Exponential of a symmetric matrix: [`expm_sym_pade`] close to zero, [`expm_sym`] otherwise.
pub fn expm_sym_fast(a: &Matrix3<f64>) -> Matrix3<f64> {
    expm_sym_pade(a).unwrap_or_else(|| expm_sym(a))
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
    fn test_pade_matches_spectral() {
        let q = rotation(Vector3::new(0.3, -1.0, 0.7), 0.9);
        let mut checked = 0;
        for scale in [0.0, 1e-9, 1e-6, 1e-3, 0.01, 0.02, 0.028] {
            for b in [
                Matrix3::identity() + scale * Matrix3::new(1.0, 0.3, -0.2, 0.3, -0.5, 0.4, -0.2, 0.4, 0.8),
                sym(&from_spectral(&Vector3::new(1.0 + scale, 1.0 + scale, 1.0 - scale), &q)),
            ] {
                let pade = half_log_pade(&b).unwrap();
                let (spectral, _, _) = half_log_spectral(&b).unwrap();
                assert!((pade - spectral).norm() <= 1e-14 * spectral.norm().max(1e-300));
                let a = 2.0 * spectral;
                assert!((expm_sym_pade(&a).unwrap() - expm_sym(&a)).norm() <= 1e-15 * expm_sym(&a).norm());
                checked += 1;
            }
        }
        assert_eq!(checked, 14);
        // outside the radius the approximants are not used
        assert!(half_log_pade(&(1.1 * Matrix3::identity())).is_none());
        assert!(expm_sym_pade(&(0.2 * Matrix3::identity())).is_none());
    }

    #[test]
    fn test_mandel_rotation() {
        use crate::mandel::{mandel_to_sym_matrix, sym_matrix_to_mandel};
        let q = rotation(Vector3::new(0.3, -1.0, 0.7), 0.9);
        let r6 = mandel_rotation(&q);
        assert!((r6.transpose() * r6 - SMatrix::<f64, 6, 6>::identity()).norm() <= 1e-14);
        let a = Matrix3::new(2.0, 0.3, -0.1, 0.3, 1.5, 0.4, -0.1, 0.4, 0.8);
        let rotated = sym_matrix_to_mandel(&(q * a * q.transpose()));
        assert!((r6 * sym_matrix_to_mandel(&a) - rotated).norm() <= 1e-14 * a.norm());
        assert!((mandel_to_sym_matrix(&(r6 * sym_matrix_to_mandel(&a))) - q * a * q.transpose()).norm() <= 1e-14 * a.norm());
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

    use crate::finite_strain::test_utils::{seeded, TensorSampling};
    use rand::Rng;

    /// Relative error, with both arguments allowed to be zero.
    fn relative_error(a: &Matrix3<f64>, reference: &Matrix3<f64>) -> f64 {
        let difference = (a - reference).norm();
        if difference == 0.0 { 0.0 } else { difference / reference.norm() }
    }

    /// Norms of the random samples: log-uniform over many orders of magnitude and, as the worst
    /// case for the approximants, uniform close to the radius.
    fn sample_norm(rng: &mut impl Rng, i: usize, radius: f64) -> f64 {
        if i % 2 == 0 {
            rng.log_uniform(1e-12 * radius, radius)
        } else {
            radius * (0.9 + 0.1 * rng.gen_range(0.0..1.0))
        }
    }

    #[test]
    fn test_pade_log_random() {
        let mut rng = seeded(1);
        let mut max_error: f64 = 0.0;
        for i in 0..20_000 {
            let norm = sample_norm(&mut rng, i, PADE_LOG_RADIUS);
            let mut x = rng.symmetric(norm);
            if i % 5 == 0 {
                // repeated eigenvalues
                let q = rng.rotation();
                let e = norm / 3f64.sqrt();
                x = sym(&from_spectral(&Vector3::new(e, e, -e), &q));
            }
            let b = Matrix3::identity() + x;
            let pade = half_log_pade(&b).expect("inside the radius");
            let (spectral, _, _) = half_log_spectral(&b).unwrap();
            max_error = max_error.max(relative_error(&pade, &spectral));
        }
        assert!(max_error <= 5e-15, "max. relative error {:e}", max_error);
    }

    #[test]
    fn test_pade_exp_random() {
        let mut rng = seeded(2);
        let mut max_error: f64 = 0.0;
        for i in 0..20_000 {
            let norm = sample_norm(&mut rng, i, PADE_EXP_RADIUS);
            let a = rng.symmetric(norm);
            let pade = expm_sym_pade(&a).expect("inside the radius");
            max_error = max_error.max(relative_error(&pade, &expm_sym(&a)));
            assert_eq!(expm_sym_fast(&a), pade);
        }
        assert!(max_error <= 2e-15, "max. relative error {:e}", max_error);
    }

    #[test]
    fn test_pade_outside_radius() {
        let mut rng = seeded(3);
        for _ in 0..1000 {
            let norm = PADE_LOG_RADIUS * (1.0 + 1e-6 + rng.gen_range(0.0..1.0));
            let x = rng.symmetric(norm);
            assert!(half_log_pade(&(Matrix3::identity() + x)).is_none());
            let norm = PADE_EXP_RADIUS * (1.0 + 1e-6 + rng.gen_range(0.0..1.0));
            let a = rng.symmetric(norm);
            assert!(expm_sym_pade(&a).is_none());
            assert_eq!(expm_sym_fast(&a), expm_sym(&a));
        }
    }

    #[test]
    fn test_sym_eigen_random() {
        let mut rng = seeded(4);
        for i in 0..20_000 {
            let scale = rng.log_uniform(1e-12, 1e3);
            let a = if i % 2 == 0 {
                rng.symmetric(scale)
            } else {
                // two eigenvalues with a relative gap between 0 and 1e-4
                let gap = if i % 4 == 1 { 0.0 } else { rng.log_uniform(1e-16, 1e-4) };
                let l = scale * (2.0 * rng.gen_range(0.0..1.0) - 1.0);
                let third = scale * rng.gen_range(0.0..1.0);
                let q = rng.rotation();
                sym(&from_spectral(&Vector3::new(l, l * (1.0 + gap), third), &q))
            };
            let (lambda, q) = sym_eigen(&a);
            assert!(relative_error(&from_spectral(&lambda, &q), &a) <= 1e-14);
            assert!((q.transpose() * q - Matrix3::identity()).norm() <= 1e-14);
        }
    }
}
