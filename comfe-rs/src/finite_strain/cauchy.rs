//! Evaluation of a [`Plasticity`] model in Cauchy stress inside a return mapping that is
//! formulated in Kirchhoff stress.
use nalgebra::SMatrix;
use nalgebra::SVector;

use crate::plasticity::{Plasticity, PlasticityState};

/// Wraps a [`Plasticity`] model so that it is evaluated at the Cauchy stress
/// $\boldsymbol\sigma = \boldsymbol\tau/J$ while the return mapping works with the Kirchhoff
/// stress $\boldsymbol\tau$. $J = \det\mathbf F$ is fixed during the return mapping.
///
/// The yield function, flow direction and hardening function of the wrapped model are
/// $\tilde f(\boldsymbol\tau,\alpha) = f(\boldsymbol\tau/J,\alpha)$,
/// $\tilde{\mathbf g} = \mathbf g(\boldsymbol\tau/J,\alpha)$ and
/// $\tilde k = k(\boldsymbol\tau/J,\alpha)$. Only the derivatives with respect to the stress
/// change, by the factor $1/J$. The flow direction is not scaled, so $\Delta\lambda\thinspace \mathbf g$
/// is still the increment of the logarithmic plastic strain and $\alpha$ keeps its meaning
/// (e.g. the equivalent plastic strain for von Mises). The elastic law stays
/// $\boldsymbol\tau = \mathbb C:\boldsymbol\varepsilon_e$.
pub struct CauchyScaled<MODEL> {
    pub inner: MODEL,
    /// $J = \det\mathbf F$
    pub volume_ratio: f64,
}

impl<MODEL> CauchyScaled<MODEL> {
    pub fn with_volume_ratio(inner: MODEL, volume_ratio: f64) -> Self {
        Self {
            inner,
            volume_ratio,
        }
    }
}

impl<
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    MODEL: Plasticity<6, N_PARAMETERS, PARAMETERS, 1>,
> Plasticity<6, N_PARAMETERS, PARAMETERS, 1> for CauchyScaled<MODEL>
{
    type Parameters = MODEL::Parameters;

    /// Creates the wrapper with $J = 1$; use [`CauchyScaled::with_volume_ratio`] instead.
    fn new(parameters: &Self::Parameters) -> Self {
        Self::with_volume_ratio(MODEL::new(parameters), 1.0)
    }

    fn set_model_state(
        &self,
        sigma: &SVector<f64, 6>,
        kappa: &SVector<f64, 1>,
        state: &mut PlasticityState<6, N_PARAMETERS, PARAMETERS, 1>,
    ) {
        let j = self.volume_ratio;
        self.inner.set_model_state(&(sigma / j), kappa, state);
        state.df_dsigma /= j;
        state.dg_dsigma /= j;
        state.dk_dsigma /= j;
    }

    fn calculate_f_only(&self, sigma: &SVector<f64, 6>, kappa: &SVector<f64, 1>) -> f64 {
        self.inner
            .calculate_f_only(&(sigma / self.volume_ratio), kappa)
    }

    fn elastic_tangent(&self) -> &SMatrix<f64, 6, 6> {
        self.inner.elastic_tangent()
    }

    fn elastic_tangent_inv(&self) -> &SMatrix<f64, 6, 6> {
        self.inner.elastic_tangent_inv()
    }

    fn characteristic_stress(&self, sigma_tr: &SVector<f64, 6>) -> f64 {
        self.inner
            .characteristic_stress(&(sigma_tr / self.volume_ratio))
    }

    fn is_admissible(&self, kappa: &SVector<f64, 1>) -> bool {
        self.inner.is_admissible(kappa)
    }
}
