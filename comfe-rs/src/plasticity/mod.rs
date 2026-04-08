pub mod general;
pub mod general_gradient_enhanced;
pub mod drucker_prager_classic;
pub mod drucker_prager_hyperbolic;
pub mod dph_damage;
pub mod isotropic_mises_plasticity;

// Re-export everything from both modules for backward compatibility
pub use general::*;
pub use general_gradient_enhanced::*;
pub use drucker_prager_classic::*;
pub use drucker_prager_hyperbolic::*;
pub use dph_damage::*;
pub use isotropic_mises_plasticity::*;