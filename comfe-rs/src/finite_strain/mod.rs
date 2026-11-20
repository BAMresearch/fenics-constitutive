//! Large-strain (total-Lagrangian) models that reuse the small-strain [`crate::plasticity`]
//! models through the Hencky kinematics.
pub mod cauchy;
pub mod hencky;
pub mod tensor;

pub use cauchy::*;
pub use hencky::*;

/// Random tensors for the tests, on top of `rand` and the `rand` integration of nalgebra.
#[cfg(test)]
pub(crate) mod test_utils {
    use nalgebra::{Matrix3, UnitQuaternion};
    use rand::distributions::{Distribution, Standard};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Reproducible generator for the given seed.
    pub fn seeded(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    pub trait TensorSampling: Rng + Sized {
        /// log-uniform in [low, high]
        fn log_uniform(&mut self, low: f64, high: f64) -> f64 {
            self.gen_range(low.ln()..=high.ln()).exp()
        }

        /// symmetric matrix with random direction and the given Frobenius norm
        fn symmetric(&mut self, norm: f64) -> Matrix3<f64> {
            let a: Matrix3<f64> = Standard.sample(self);
            let a = a.map(|value| 2.0 * value - 1.0);
            let a = 0.5 * (a + a.transpose());
            a * (norm / a.norm())
        }

        /// rotation, uniformly distributed over SO(3)
        fn rotation(&mut self) -> Matrix3<f64> {
            let q: UnitQuaternion<f64> = Standard.sample(self);
            q.to_rotation_matrix().into_inner()
        }
    }

    impl<R: Rng> TensorSampling for R {}
}
