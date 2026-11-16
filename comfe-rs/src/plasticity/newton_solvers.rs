use nalgebra::{Const, DMatrix, DimMin, SMatrix, SVector};

use crate::plasticity::{Plasticity, PlasticityState};

/// Newton solver with backtracking line search for the return mapping, formulated in the
/// unknowns $x = (\Delta\varepsilon_p, \Delta\lambda, \kappa)$ with $\sigma = \sigma_{tr} - C\Delta\varepsilon_p$.
///
/// The residual is
/// $r_\varepsilon = \Delta\varepsilon_p - \Delta\lambda\, g$, $r_f = f / \sigma_c$,
/// $r_\kappa = \kappa - \kappa_0 - \Delta\lambda\, k$,
/// where $\sigma_c$ is the characteristic stress of the model. All residuals are therefore
/// dimensionless and are measured with one norm, both for the convergence check
/// $\lVert r\rVert \le atol + rtol \lVert r_0\rVert$ and for the line search.
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
}

pub struct SolverResult<const STRESS_STRAIN: usize, const KAPPA: usize> {
    pub sigma: SVector<f64, STRESS_STRAIN>,
    pub del_plastic_strain: SVector<f64, STRESS_STRAIN>,
    pub kappa: SVector<f64, KAPPA>,
    pub del_lambda: f64,
    pub iterations: usize,
}

/// Convergence history of a failed return mapping. Residuals are the scaled residuals of
/// [`NewtonSolverStrain`]; increments are the accepted (damped) Newton updates.
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

#[derive(Default)]
struct ConvergenceHistory {
    eps_res: Vec<f64>,
    f_res: Vec<f64>,
    kappa_res: Vec<f64>,
    eps_incr: Vec<f64>,
    lambda_incr: Vec<f64>,
    kappa_incr: Vec<f64>,
}

impl ConvergenceHistory {
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

    fn into_error<const N: usize>(self, dres: &SMatrix<f64, N, N>) -> SolverError {
        // only evaluated on failure, so the copy into a dynamic matrix does not matter
        let mut dres_dyn = DMatrix::<f64>::zeros(N, N);
        dres_dyn.copy_from(dres);
        let svd = dres_dyn.svd_unordered(false, false);
        SolverError {
            eps_error_incr: self.eps_incr,
            eps_error_res: self.eps_res,
            lambda_error_incr: self.lambda_incr,
            f_error_res: self.f_res,
            kappa_error_incr: self.kappa_incr,
            kappa_error_res: self.kappa_res,
            singular_values: svd.singular_values.as_slice().to_vec(),
            condition: svd.singular_values.max() / svd.singular_values.min(),
        }
    }
}

/// Fills `dres` with the Jacobian of the stress-based residual
/// $r_\sigma = \sigma - \sigma_{tr} + \Delta\lambda\, C g$, $r_f = f$,
/// $r_\kappa = \kappa - \kappa_0 - \Delta\lambda\, k$ with respect to
/// $(\sigma, \Delta\lambda, \kappa)$, evaluated at `state`. Its inverse yields the
/// consistent tangent after the return mapping has converged.
pub fn stress_based_jacobian<
    const N: usize,
    const STRESS_STRAIN: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    const KAPPA: usize,
>(
    elastic_tangent: &SMatrix<f64, STRESS_STRAIN, STRESS_STRAIN>,
    state: &PlasticityState<STRESS_STRAIN, N_PARAMETERS, PARAMETERS, KAPPA>,
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
pub fn strain_based_jacobian<
    const N: usize,
    const STRESS_STRAIN: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    const KAPPA: usize,
>(
    elastic_tangent: &SMatrix<f64, STRESS_STRAIN, STRESS_STRAIN>,
    state: &PlasticityState<STRESS_STRAIN, N_PARAMETERS, PARAMETERS, KAPPA>,
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
    pub fn new(model: &'a T, atol: f64, rtol: f64, maxit: usize) -> Self {
        assert_eq!(N, STRESS_STRAIN + 1 + KAPPA, "Incompatible const generic argument.");
        Self {
            atol,
            rtol,
            maxit,
            max_line_search: 20,
            model,
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
        state: &mut PlasticityState<STRESS_STRAIN, N_PARAMETERS, PARAMETERS, KAPPA>,
    ) -> (SVector<f64, N>, SVector<f64, STRESS_STRAIN>) {
        let del_eps_pl: SVector<f64, STRESS_STRAIN> = sol.fixed_rows::<STRESS_STRAIN>(0).into();
        let del_lambda = sol[STRESS_STRAIN];
        let kappa: SVector<f64, KAPPA> = sol.fixed_rows::<KAPPA>(STRESS_STRAIN + 1).into();

        let sigma = sigma_tr - self.model.elastic_tangent() * &del_eps_pl;
        self.model.set_model_state(&sigma, &kappa, state);

        let mut res = SVector::<f64, N>::zeros();
        res.fixed_rows_mut::<STRESS_STRAIN>(0)
            .copy_from(&(del_eps_pl - state.g * del_lambda));
        res[STRESS_STRAIN] = state.f * f_scale;
        res.fixed_rows_mut::<KAPPA>(STRESS_STRAIN + 1)
            .copy_from(&(kappa - kappa_0 - state.k * del_lambda));
        (res, sigma)
    }

    /// Solves the return mapping starting from the trial state. On success, `state` holds the
    /// plasticity state at the converged stress and kappa.
    pub fn solve(
        &self,
        sigma_tr: &SVector<f64, STRESS_STRAIN>,
        kappa_0: &SVector<f64, KAPPA>,
        state: &mut PlasticityState<STRESS_STRAIN, N_PARAMETERS, PARAMETERS, KAPPA>,
    ) -> Result<SolverResult<STRESS_STRAIN, KAPPA>, SolverError> {
        let elastic_tangent = self.model.elastic_tangent();
        let f_scale = 1.0 / self.model.characteristic_stress(sigma_tr);

        let mut sol = SVector::<f64, N>::zeros();
        sol.fixed_rows_mut::<KAPPA>(STRESS_STRAIN + 1)
            .copy_from(kappa_0);
        let (mut res, mut sigma) = self.residual(&sol, sigma_tr, kappa_0, f_scale, state);
        let tolerance = self.atol + self.rtol * res.norm();

        let mut dres = SMatrix::<f64, N, N>::zeros();
        let mut history = ConvergenceHistory::default();
        history.push_residual::<N, STRESS_STRAIN, KAPPA>(&res);

        let mut iterations = 0;
        while res.norm() > tolerance {
            if iterations >= self.maxit {
                return Err(history.into_error(&dres));
            }
            iterations += 1;

            strain_based_jacobian(elastic_tangent, state, &mut dres, sol[STRESS_STRAIN], f_scale);
            let Some(dx) = dres.lu().solve(&res) else {
                return Err(history.into_error(&dres));
            };
            // A full Newton step at the level of round-off: the residual cannot decrease any
            // further, so the step is accepted and the iteration stops.
            let converged_incr = dx.norm() <= self.atol + self.rtol * sol.norm();

            // Backtracking line search: plain Newton cycles without converging when the
            // return maps far into the (damage-shrunk) yield surface, e.g. near the hyperbola
            // apex. Trial points with a negative plastic multiplier or a kappa that the model
            // rejects are not evaluated.
            let res_norm_prev = res.norm();
            let mut alpha = 1.0;
            let mut accepted = false;
            for _ in 0..self.max_line_search {
                let trial = sol - alpha * &dx;
                let trial_kappa: SVector<f64, KAPPA> =
                    trial.fixed_rows::<KAPPA>(STRESS_STRAIN + 1).into();
                if trial[STRESS_STRAIN] >= 0.0 && self.model.is_admissible(&trial_kappa) {
                    let (trial_res, trial_sigma) =
                        self.residual(&trial, sigma_tr, kappa_0, f_scale, state);
                    if converged_incr
                        || trial_res.norm() <= (1.0 - 1e-4 * alpha) * res_norm_prev
                    {
                        history.push_increment::<N, STRESS_STRAIN, KAPPA>(&(trial - sol));
                        history.push_residual::<N, STRESS_STRAIN, KAPPA>(&trial_res);
                        sol = trial;
                        res = trial_res;
                        sigma = trial_sigma;
                        accepted = true;
                        break;
                    }
                }
                alpha *= 0.5;
            }
            if !accepted {
                return Err(history.into_error(&dres));
            }
            if converged_incr {
                break;
            }
        }

        Ok(SolverResult {
            sigma,
            del_plastic_strain: sol.fixed_rows::<STRESS_STRAIN>(0).into(),
            kappa: sol.fixed_rows::<KAPPA>(STRESS_STRAIN + 1).into(),
            del_lambda: sol[STRESS_STRAIN],
            iterations,
        })
    }
}
