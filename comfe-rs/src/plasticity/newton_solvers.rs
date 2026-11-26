use nalgebra::{Const, DMatrix, DimMin, SMatrix, SVector};

use crate::plasticity::{Plasticity, PlasticityState};

/// Factor between the first-order estimate of the round-off level of the residual and the
/// residual that is accepted as converged.
const ROUNDOFF_SAFETY: f64 = 10.0;

/// Newton solver with backtracking line search for the return mapping, formulated in the
/// unknowns $x = (\Delta\varepsilon_p, \Delta\lambda, \kappa)$ with $\sigma = \sigma_{tr} - C\Delta\varepsilon_p$.
///
/// The residual is
/// $r_\varepsilon = \Delta\varepsilon_p - \Delta\lambda\thinspace g$, $r_f = f / \sigma_c$,
/// $r_\kappa = \kappa - \kappa_0 - \Delta\lambda\thinspace k$,
/// where $\sigma_c$ is the characteristic stress of the model. All residuals are therefore
/// dimensionless and are measured with one norm, both for the convergence check
/// $\lVert r\rVert \le atol + rtol \lVert r_0\rVert$ and for the line search. The iteration
/// also stops when the residual is below its estimated round-off level, see
/// `roundoff_level`, provided that $r_f$ is below the tolerance.
///
/// # Potential improvements
/// The round-off level is high near the apex of the hyperbolic Drucker-Prager model because of
/// two subtractions of nearly equal numbers: the stress $\sigma = \sigma_{tr} - C\Delta\varepsilon_p$
/// with $\lVert\sigma\rVert \ll \lVert\sigma_{tr}\rVert$, and the deviator
/// $s = \sigma - \tfrac{1}{3}I_1 I$ with $\lVert s\rVert \ll \lVert\sigma\rVert$, which the model
/// functions compute. The curvature $\partial g/\partial\sigma \sim 1/(bd)$ amplifies their
/// round-off errors. With $\sigma$ as unknown, only the first subtraction disappears: in tests,
/// the noise of the residual decreased by a factor of 2 to 4500, but remained above `atol` for
/// $d \lesssim 10^{-3}$.
///
/// For isotropic models whose flow potential depends on the deviator only through $J_2$ (von
/// Mises, Drucker-Prager), the deviator of the solution is parallel to the trial deviator. A
/// return mapping in the unknowns $(I_1, \varphi, \Delta\lambda, \kappa)$ with $s = \varphi s_{tr}$,
/// and with model functions of $I_1$ and $s$ instead of $\sigma$, would
/// - represent small deviators with full relative precision, which likely makes the round-off
///   handling unnecessary,
/// - reduce the Newton system from 8 to 4 unknowns, and
/// - allow enforcing $0 \le \varphi \le 1$ in the line search, so that the deviator cannot reverse
///   its direction. This is how the iteration fails for returns deep into the apex region (see
///   [`DruckerPragerHyperbolic3D`](crate::plasticity::DruckerPragerHyperbolic3D)), so the
///   bound could remove that limitation without a separate case for the apex.
///
/// It requires the model functions in terms of $(I_1, s)$ and does not apply to models that
/// depend on the Lode angle or are anisotropic, for which the current formulation would remain.
/// This has not been tested yet.
pub struct NewtonSolverStrain<
    'a,
    const N: usize, //TODO:fix when const generics allow arithmetics
    const STRESS_STRAIN: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    const KAPPA: usize,
    T: Plasticity<STRESS_STRAIN, N_PARAMETERS, PARAMETERS, KAPPA>,
> {
    pub atol: f64,
    pub rtol: f64,
    pub maxit: usize,
    /// Maximum number of step halvings in the line search before the solver fails.
    pub max_line_search: usize,
    pub model: &'a T,
    /// Elastic tangent $C$, assumed constant during the return mapping.
    pub elastic_tangent: &'a SMatrix<f64, STRESS_STRAIN, STRESS_STRAIN>,
}

pub struct SolverResult<const STRESS_STRAIN: usize, const KAPPA: usize> {
    pub sigma: SVector<f64, STRESS_STRAIN>,
    pub del_plastic_strain: SVector<f64, STRESS_STRAIN>,
    pub kappa: SVector<f64, KAPPA>,
    pub del_lambda: f64,
}

/// Convergence history of a failed return mapping. Residuals are the scaled residuals of
/// [`NewtonSolverStrain`]; increments are the accepted (damped) Newton updates. The singular
/// values and the condition number are those of the last Newton matrix; they are empty and NaN
/// if that matrix has non-finite entries.
#[derive(Debug)]
pub struct SolverError {
    pub eps_error_incr: Vec<f64>,
    pub eps_error_res: Vec<f64>,
    pub lambda_error_incr: Vec<f64>,
    pub f_error_res: Vec<f64>,
    pub kappa_error_incr: Vec<f64>,
    pub kappa_error_res: Vec<f64>,
    pub singular_values: Vec<f64>,
    pub condition: f64,
}

/// Records the convergence history of the Newton iteration, see [`NewtonSolverStrain::solve`].
trait Recorder {
    fn push_residual<const N: usize, const STRESS_STRAIN: usize, const KAPPA: usize>(
        &mut self,
        res: &SVector<f64, N>,
    );
    fn push_increment<const N: usize, const STRESS_STRAIN: usize, const KAPPA: usize>(
        &mut self,
        incr: &SVector<f64, N>,
    );
}

/// Records nothing.
impl Recorder for () {
    #[inline]
    fn push_residual<const N: usize, const STRESS_STRAIN: usize, const KAPPA: usize>(
        &mut self,
        _res: &SVector<f64, N>,
    ) {
    }
    #[inline]
    fn push_increment<const N: usize, const STRESS_STRAIN: usize, const KAPPA: usize>(
        &mut self,
        _incr: &SVector<f64, N>,
    ) {
    }
}

#[derive(Default)]
struct ConvergenceHistory {
    eps_res: Vec<f64>,
    f_res: Vec<f64>,
    kappa_res: Vec<f64>,
    eps_incr: Vec<f64>,
    lambda_incr: Vec<f64>,
    kappa_incr: Vec<f64>,
}

impl Recorder for ConvergenceHistory {
    fn push_residual<const N: usize, const STRESS_STRAIN: usize, const KAPPA: usize>(
        &mut self,
        res: &SVector<f64, N>,
    ) {
        self.eps_res.push(res.fixed_rows::<STRESS_STRAIN>(0).norm());
        self.f_res.push(res[STRESS_STRAIN].abs());
        self.kappa_res
            .push(res.fixed_rows::<KAPPA>(STRESS_STRAIN + 1).norm());
    }

    fn push_increment<const N: usize, const STRESS_STRAIN: usize, const KAPPA: usize>(
        &mut self,
        incr: &SVector<f64, N>,
    ) {
        self.eps_incr.push(incr.fixed_rows::<STRESS_STRAIN>(0).norm());
        self.lambda_incr.push(incr[STRESS_STRAIN].abs());
        self.kappa_incr
            .push(incr.fixed_rows::<KAPPA>(STRESS_STRAIN + 1).norm());
    }
}

impl ConvergenceHistory {
    fn into_error<const N: usize>(self, dres: &SMatrix<f64, N, N>) -> SolverError {
        // The SVD does not terminate on a matrix with NaN or infinite entries, which occur
        // when the model state is not defined, e.g. dk/dsigma for a vanishing flow direction.
        let (singular_values, condition) = if dres.iter().all(|x| x.is_finite()) {
            // only evaluated on failure, so the copy into a dynamic matrix does not matter
            let mut dres_dyn = DMatrix::<f64>::zeros(N, N);
            dres_dyn.copy_from(dres);
            let svd = dres_dyn.svd_unordered(false, false);
            (
                svd.singular_values.as_slice().to_vec(),
                svd.singular_values.max() / svd.singular_values.min(),
            )
        } else {
            (Vec::new(), f64::NAN)
        };
        SolverError {
            eps_error_incr: self.eps_incr,
            eps_error_res: self.eps_res,
            lambda_error_incr: self.lambda_incr,
            f_error_res: self.f_res,
            kappa_error_incr: self.kappa_incr,
            kappa_error_res: self.kappa_res,
            singular_values,
            condition,
        }
    }
}

/// Fills `dres` with the Jacobian of the stress-based residual
/// $r_\sigma = \sigma - \sigma_{tr} + \Delta\lambda\thinspace C g$, $r_f = f$,
/// $r_\kappa = \kappa - \kappa_0 - \Delta\lambda\thinspace k$ with respect to
/// $(\sigma, \Delta\lambda, \kappa)$, evaluated at `state`. Its inverse yields the
/// consistent tangent after the return mapping has converged.
pub fn stress_based_jacobian<const N: usize, const STRESS_STRAIN: usize, const KAPPA: usize>(
    elastic_tangent: &SMatrix<f64, STRESS_STRAIN, STRESS_STRAIN>,
    state: &PlasticityState<STRESS_STRAIN, KAPPA>,
    dres: &mut SMatrix<f64, N, N>,
    del_lambda: f64,
) {
    assert!(N == STRESS_STRAIN + KAPPA + 1);
    // dres_sigma_dsigma
    dres.fixed_view_mut::<STRESS_STRAIN, STRESS_STRAIN>(0, 0)
        .copy_from(
            &(SMatrix::<f64, STRESS_STRAIN, STRESS_STRAIN>::identity()
                + elastic_tangent * state.dg_dsigma * del_lambda),
        );
    // dres_sigma_dlambda
    dres.fixed_view_mut::<STRESS_STRAIN, 1>(0, STRESS_STRAIN)
        .copy_from(&(elastic_tangent * state.g));
    // dres_sigma_dkappa
    dres.fixed_view_mut::<STRESS_STRAIN, KAPPA>(0, STRESS_STRAIN + 1)
        .copy_from(&(elastic_tangent * state.dg_dkappa * del_lambda));

    // dres_f_dsigma
    dres.fixed_view_mut::<1, STRESS_STRAIN>(STRESS_STRAIN, 0)
        .copy_from(&state.df_dsigma);
    // dres_f_dlambda
    dres[(STRESS_STRAIN, STRESS_STRAIN)] = 0.0;
    // dres_f_dkappa
    dres.fixed_view_mut::<1, KAPPA>(STRESS_STRAIN, STRESS_STRAIN + 1)
        .copy_from(&state.df_dkappa);

    // dres_kappa_dsigma
    dres.fixed_view_mut::<KAPPA, STRESS_STRAIN>(STRESS_STRAIN + 1, 0)
        .copy_from(&(state.dk_dsigma * (-del_lambda)));
    // dres_kappa_dlambda
    dres.fixed_view_mut::<KAPPA, 1>(STRESS_STRAIN + 1, STRESS_STRAIN)
        .copy_from(&(-state.k));
    // dres_kappa_dkappa
    dres.fixed_view_mut::<KAPPA, KAPPA>(STRESS_STRAIN + 1, STRESS_STRAIN + 1)
        .copy_from(
            &(SMatrix::<f64, KAPPA, KAPPA>::identity() - state.dk_dkappa * del_lambda),
        );
}

/// Fills `dres` with the Jacobian of the residual of [`NewtonSolverStrain`] with respect to
/// $(\Delta\varepsilon_p, \Delta\lambda, \kappa)$, evaluated at `state`. `f_scale` is
/// $1/\sigma_c$, which scales the yield function row.
pub fn strain_based_jacobian<const N: usize, const STRESS_STRAIN: usize, const KAPPA: usize>(
    elastic_tangent: &SMatrix<f64, STRESS_STRAIN, STRESS_STRAIN>,
    state: &PlasticityState<STRESS_STRAIN, KAPPA>,
    dres: &mut SMatrix<f64, N, N>,
    del_lambda: f64,
    f_scale: f64,
) {
    assert!(N == STRESS_STRAIN + KAPPA + 1);
    // dres_eps_deps
    dres.fixed_view_mut::<STRESS_STRAIN, STRESS_STRAIN>(0, 0)
        .copy_from(
            &(SMatrix::<f64, STRESS_STRAIN, STRESS_STRAIN>::identity()
                + state.dg_dsigma * elastic_tangent * del_lambda),
        );
    // dres_eps_dlambda
    dres.fixed_view_mut::<STRESS_STRAIN, 1>(0, STRESS_STRAIN)
        .copy_from(&(-state.g));
    // dres_eps_dkappa
    dres.fixed_view_mut::<STRESS_STRAIN, KAPPA>(0, STRESS_STRAIN + 1)
        .copy_from(&(state.dg_dkappa * (-del_lambda)));

    // dres_f_deps
    dres.fixed_view_mut::<1, STRESS_STRAIN>(STRESS_STRAIN, 0)
        .copy_from(&(-(state.df_dsigma * elastic_tangent)));
    // dres_f_dlambda
    dres[(STRESS_STRAIN, STRESS_STRAIN)] = 0.0;
    // dres_f_dkappa
    dres.fixed_view_mut::<1, KAPPA>(STRESS_STRAIN, STRESS_STRAIN + 1)
        .copy_from(&state.df_dkappa);

    // dres_kappa_deps
    dres.fixed_view_mut::<KAPPA, STRESS_STRAIN>(STRESS_STRAIN + 1, 0)
        .copy_from(&(state.dk_dsigma * elastic_tangent * del_lambda));
    // dres_kappa_dlambda
    dres.fixed_view_mut::<KAPPA, 1>(STRESS_STRAIN + 1, STRESS_STRAIN)
        .copy_from(&(-state.k));
    // dres_kappa_dkappa
    dres.fixed_view_mut::<KAPPA, KAPPA>(STRESS_STRAIN + 1, STRESS_STRAIN + 1)
        .copy_from(
            &(SMatrix::<f64, KAPPA, KAPPA>::identity() - state.dk_dkappa * del_lambda),
        );
    dres.fixed_view_mut::<1, N>(STRESS_STRAIN, 0)
        .scale_mut(f_scale);
}

impl<
    'a,
    const N: usize,
    const STRESS_STRAIN: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    const KAPPA: usize,
    T: Plasticity<STRESS_STRAIN, N_PARAMETERS, PARAMETERS, KAPPA>,
> NewtonSolverStrain<'a, N, STRESS_STRAIN, N_PARAMETERS, PARAMETERS, KAPPA, T>
where
    // required for the LU decomposition of the static N x N Newton matrix
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    pub fn new(
        model: &'a T,
        elastic_tangent: &'a SMatrix<f64, STRESS_STRAIN, STRESS_STRAIN>,
        atol: f64,
        rtol: f64,
        maxit: usize,
    ) -> Self {
        assert_eq!(N, STRESS_STRAIN + 1 + KAPPA, "Incompatible const generic argument.");
        Self {
            atol,
            rtol,
            maxit,
            max_line_search: 20,
            model,
            elastic_tangent,
        }
    }

    /// Evaluates the scaled residual at `sol` and sets `state` to the plasticity state at the
    /// corresponding stress and kappa. Returns the residual and the stress.
    fn residual(
        &self,
        sol: &SVector<f64, N>,
        sigma_tr: &SVector<f64, STRESS_STRAIN>,
        kappa_0: &SVector<f64, KAPPA>,
        f_scale: f64,
        state: &mut PlasticityState<STRESS_STRAIN, KAPPA>,
    ) -> (SVector<f64, N>, SVector<f64, STRESS_STRAIN>) {
        let del_eps_pl: SVector<f64, STRESS_STRAIN> = sol.fixed_rows::<STRESS_STRAIN>(0).into();
        let del_lambda = sol[STRESS_STRAIN];
        let kappa: SVector<f64, KAPPA> = sol.fixed_rows::<KAPPA>(STRESS_STRAIN + 1).into();

        let sigma = sigma_tr - self.elastic_tangent * &del_eps_pl;
        self.model.set_model_state(&sigma, &kappa, state);

        let mut res = SVector::<f64, N>::zeros();
        res.fixed_rows_mut::<STRESS_STRAIN>(0)
            .copy_from(&(del_eps_pl - state.g * del_lambda));
        res[STRESS_STRAIN] = state.f * f_scale;
        res.fixed_rows_mut::<KAPPA>(STRESS_STRAIN + 1)
            .copy_from(&(kappa - kappa_0 - state.k * del_lambda));
        (res, sigma)
    }

    /// Estimated round-off level of the residual. The stress
    /// $\sigma = \sigma_{tr} - C\Delta\varepsilon_p$ carries an error of about
    /// $\epsilon(\lVert\sigma_{tr}\rVert + \lVert\sigma\rVert)$ due to cancellation, which the
    /// residual amplifies by $\lVert\partial r/\partial\sigma\rVert$. Near a sharply curved part
    /// of the yield surface (e.g. the apex of the hyperbolic Drucker-Prager model), this
    /// round-off level of $r_\varepsilon$ can be orders of magnitude above `atol`. `state` must
    /// be the plasticity state at `sigma`.
    fn roundoff_level(
        sigma_tr: &SVector<f64, STRESS_STRAIN>,
        sigma: &SVector<f64, STRESS_STRAIN>,
        del_lambda: f64,
        f_scale: f64,
        state: &PlasticityState<STRESS_STRAIN, KAPPA>,
    ) -> f64 {
        let del_sigma = ROUNDOFF_SAFETY * f64::EPSILON * (sigma_tr.norm() + sigma.norm());
        let dres_dsigma_norm = ((del_lambda * del_lambda)
            * (state.dg_dsigma.norm_squared() + state.dk_dsigma.norm_squared())
            + (f_scale * f_scale) * state.df_dsigma.norm_squared())
        .sqrt();
        del_sigma * dres_dsigma_norm
    }

    /// Solves the return mapping starting from the trial state. On success, `state` holds the
    /// plasticity state at the converged stress and kappa.
    ///
    /// The convergence history is only needed to report a failure, but recording it allocates
    /// at every plastic quadrature point. The iteration therefore runs without recording, and a
    /// failed solve is repeated with recording; being deterministic, it fails in the same way.
    pub fn solve(
        &self,
        sigma_tr: &SVector<f64, STRESS_STRAIN>,
        kappa_0: &SVector<f64, KAPPA>,
        state: &mut PlasticityState<STRESS_STRAIN, KAPPA>,
    ) -> Result<SolverResult<STRESS_STRAIN, KAPPA>, SolverError> {
        self.iterate(sigma_tr, kappa_0, state, &mut ()).or_else(|_| {
            let mut history = ConvergenceHistory::default();
            self.iterate(sigma_tr, kappa_0, state, &mut history)
                .map_err(|dres| history.into_error(&dres))
        })
    }

    /// Newton iteration of [`Self::solve`]. On failure, returns the last Newton matrix.
    fn iterate<R: Recorder>(
        &self,
        sigma_tr: &SVector<f64, STRESS_STRAIN>,
        kappa_0: &SVector<f64, KAPPA>,
        state: &mut PlasticityState<STRESS_STRAIN, KAPPA>,
        recorder: &mut R,
    ) -> Result<SolverResult<STRESS_STRAIN, KAPPA>, SMatrix<f64, N, N>> {
        let elastic_tangent = self.elastic_tangent;
        let f_scale = 1.0 / self.model.characteristic_stress(sigma_tr);

        let mut sol = SVector::<f64, N>::zeros();
        sol.fixed_rows_mut::<KAPPA>(STRESS_STRAIN + 1)
            .copy_from(kappa_0);
        let (mut res, mut sigma) = self.residual(&sol, sigma_tr, kappa_0, f_scale, state);
        let tolerance = self.atol + self.rtol * res.norm();

        let mut dres = SMatrix::<f64, N, N>::zeros();
        recorder.push_residual::<N, STRESS_STRAIN, KAPPA>(&res);

        let mut iterations = 0;
        // Converged if the residual is below the tolerance, or below its round-off level. The
        // yield function is not amplified like the other residuals, so it must still be below
        // the tolerance; otherwise a large round-off level of r_eps could hide a stress that is
        // not on the yield surface.
        let mut roundoff = Self::roundoff_level(sigma_tr, &sigma, sol[STRESS_STRAIN], f_scale, state);
        while res.norm() > tolerance
            && !(res.norm() <= tolerance + roundoff && res[STRESS_STRAIN].abs() <= tolerance)
        {
            if iterations >= self.maxit {
                return Err(dres);
            }
            iterations += 1;

            strain_based_jacobian(elastic_tangent, state, &mut dres, sol[STRESS_STRAIN], f_scale);
            let Some(dx) = dres.lu().solve(&res) else {
                return Err(dres);
            };
            // A full Newton step at the level of round-off is accepted without a sufficient
            // decrease of the residual, which cannot decrease reliably any further. Whether the
            // iteration has converged is still decided by the residual.
            let roundoff_step = dx.norm() <= self.atol + self.rtol * sol.norm();

            // Backtracking line search: plain Newton cycles without converging when the
            // return maps far into the (damage-shrunk) yield surface, e.g. near the hyperbola
            // apex. Trial points with a negative plastic multiplier or a stress or kappa that
            // the model rejects are not evaluated.
            let res_norm_prev = res.norm();
            let mut alpha = 1.0;
            let mut accepted = false;
            for _ in 0..self.max_line_search {
                let trial = sol - alpha * &dx;
                let trial_eps: SVector<f64, STRESS_STRAIN> =
                    trial.fixed_rows::<STRESS_STRAIN>(0).into();
                let trial_kappa: SVector<f64, KAPPA> =
                    trial.fixed_rows::<KAPPA>(STRESS_STRAIN + 1).into();
                if trial[STRESS_STRAIN] >= 0.0
                    && self.model.is_admissible(
                        &(sigma_tr - elastic_tangent * &trial_eps),
                        &trial_kappa,
                    )
                {
                    let (trial_res, trial_sigma) =
                        self.residual(&trial, sigma_tr, kappa_0, f_scale, state);
                    let trial_roundoff = Self::roundoff_level(
                        sigma_tr, &trial_sigma, trial[STRESS_STRAIN], f_scale, state,
                    );
                    // Within the round-off level, the residual cannot decrease reliably any
                    // further, and only the yield function still has to decrease.
                    let roundoff_decrease = trial_res.norm() <= tolerance + trial_roundoff
                        && trial_res[STRESS_STRAIN].abs()
                            <= (1.0 - 1e-4 * alpha) * res[STRESS_STRAIN].abs();
                    if (roundoff_step && alpha == 1.0)
                        || roundoff_decrease
                        || trial_res.norm() <= (1.0 - 1e-4 * alpha) * res_norm_prev
                    {
                        recorder.push_increment::<N, STRESS_STRAIN, KAPPA>(&(trial - sol));
                        recorder.push_residual::<N, STRESS_STRAIN, KAPPA>(&trial_res);
                        sol = trial;
                        res = trial_res;
                        sigma = trial_sigma;
                        roundoff = trial_roundoff;
                        accepted = true;
                        break;
                    }
                }
                alpha *= 0.5;
            }
            if !accepted {
                return Err(dres);
            }
        }

        Ok(SolverResult {
            sigma,
            del_plastic_strain: sol.fixed_rows::<STRESS_STRAIN>(0).into(),
            kappa: sol.fixed_rows::<KAPPA>(STRESS_STRAIN + 1).into(),
            del_lambda: sol[STRESS_STRAIN],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_into_error_with_non_finite_matrix() {
        let mut dres = SMatrix::<f64, 8, 8>::identity();
        dres[(6, 7)] = f64::NAN;
        let error = ConvergenceHistory::default().into_error(&dres);
        assert!(error.singular_values.is_empty());
        assert!(error.condition.is_nan());
    }
}
