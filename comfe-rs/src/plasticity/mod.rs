pub mod general;
pub mod drucker_prager_classic;
pub mod drucker_prager_hyperbolic;
pub mod isotropic_mises_plasticity;

// Re-export everything from both modules for backward compatibility
pub use general::*;
pub use drucker_prager_classic::*;
pub use drucker_prager_hyperbolic::*;
pub use isotropic_mises_plasticity::*;