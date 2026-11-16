pub mod general;
//pub mod general_gradient_enhanced;
pub mod drucker_prager_classic;
pub mod drucker_prager_hyperbolic;
pub mod isotropic_mises_plasticity;
pub mod newton_solvers;

// Re-export everything from both modules for backward compatibility
pub use general::*;
//pub use general_gradient_enhanced::*;
pub use drucker_prager_classic::*;
pub use drucker_prager_hyperbolic::*;
pub use isotropic_mises_plasticity::*;
pub use newton_solvers::*;
