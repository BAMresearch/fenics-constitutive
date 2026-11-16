//! Large-strain (total-Lagrangian) models that reuse the small-strain [`crate::plasticity`]
//! models through the Hencky kinematics.
pub mod cauchy;
pub mod hencky;
pub mod tensor;

pub use cauchy::*;
pub use hencky::*;
