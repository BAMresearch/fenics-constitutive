use nalgebra::{DMatrix, LU, RowSVector, SMatrix, SVector};

use crate::{consts::id, plasticity::Plasticity};
pub struct NewtonSolver<
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
    pub model: &'a mut T,
}
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
    pub model: &'a mut T,
}

pub struct NewtonSchurComplementSolver<
    'a,
    const N: usize, //TODO:fix when const generics allow arithmetics
    const STRESS_STRAIN: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    const KAPPA: usize,
    const KAPPA_P1: usize,
    T: Plasticity<STRESS_STRAIN, N_PARAMETERS, PARAMETERS, KAPPA>,
> {
    pub atol: f64,
    pub rtol: f64,
    pub maxit: usize,
    pub model: &'a mut T,
}

pub struct SolverResult<const STRESS_STRAIN: usize, const KAPPA: usize> {
    pub sigma: SVector<f64, STRESS_STRAIN>,
    pub kappa: SVector<f64, KAPPA>,
    pub del_lambda: f64,
    pub iterations: usize,
}
#[derive(Debug)]
pub struct SolverError {
    pub sigma_error_incr: Vec<f64>,
    pub sigma_error_res: Vec<f64>,
    pub lambda_error_incr: Vec<f64>,
    pub f_error_res: Vec<f64>,
    pub kappa_error_incr: Vec<f64>,
    pub kappa_error_res: Vec<f64>,
    pub singular_values: Vec<f64>,
    pub condition: f64,
}
#[derive(Debug)]
pub struct SchurComplementSolverError {
    pub sigma_error_incr: Vec<f64>,
    pub sigma_error_res: Vec<f64>,
    pub lambda_error_incr: Vec<f64>,
    pub f_error_res: Vec<f64>,
    pub kappa_error_incr: Vec<f64>,
    pub kappa_error_res: Vec<f64>,
    //pub condition_schur_complement: f64,
    //pub condition_jss: f64,
}

impl<
    'a,
    const N: usize,
    const STRESS_STRAIN: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    const KAPPA: usize,
    T: Plasticity<STRESS_STRAIN, N_PARAMETERS, PARAMETERS, KAPPA>,
> NewtonSolver<'a, N, STRESS_STRAIN, N_PARAMETERS, PARAMETERS, KAPPA, T>
{
    pub fn new(model: &'a mut T, atol: f64, rtol: f64, maxit: usize) -> Self {
        assert_eq!(N, STRESS_STRAIN + 1 + KAPPA);
        Self {
            atol,
            rtol,
            maxit,
            model,
        }
    }

    pub fn solve(
        &mut self,
        sigma_tr: &SVector<f64, STRESS_STRAIN>,
        kappa_0: &SVector<f64, KAPPA>,
    ) -> Result<SolverResult<STRESS_STRAIN, KAPPA>, SolverError> {
        let mut del_lambda = 0.0;
        let mut sigma: SVector<f64, STRESS_STRAIN>;
        let mut kappa: SVector<f64, KAPPA>;
        let mut sol_0 = SVector::<f64, N>::zeros();
        let mut sol_1 = SVector::<f64, N>::zeros();
        sol_1.fixed_rows_mut::<STRESS_STRAIN>(0).copy_from(sigma_tr);
        sol_1.fixed_rows_mut::<1>(STRESS_STRAIN).x = 0.0;
        sol_1
            .fixed_rows_mut::<KAPPA>(STRESS_STRAIN + 1)
            .copy_from(kappa_0);

        let scaling_factors = self.model.residual_scaling(sigma_tr, kappa_0);

        let mut res_sigma = SVector::<f64, STRESS_STRAIN>::zeros();
        let mut res_kappa = SVector::<f64, KAPPA>::zeros();
        let mut res_f = self.model.f() * scaling_factors.1;

        let mut dres = SMatrix::<f64, N, N>::zeros();

        let mut res = SVector::<f64, N>::zeros();
        res[STRESS_STRAIN] = res_f;

        let mut i = 0;

        let mut sigma_prev: SVector<f64, STRESS_STRAIN>;
        let mut kappa_prev: SVector<f64, KAPPA>;
        let mut del_lambda_prev: f64;
        let mut res_sigma_norm: Vec<f64> = vec![res_sigma.norm()];
        let mut res_kappa_norm: Vec<f64> = vec![res_kappa.norm()];
        let mut res_f_norm: Vec<f64> = vec![res_f];
        let mut sigma_error_incr: Vec<f64> = vec![0.0];
        let mut kappa_error_incr: Vec<f64> = vec![0.0];
        let mut lambda_error_incr: Vec<f64> = vec![0.0];
        loop {
            sol_0 = sol_1;

            //Fill the newton matrix
            self.update_newton_matrix(&mut dres, del_lambda, scaling_factors);

            //this part of the code is ugly
            let mut dres_dyn = DMatrix::<f64>::zeros(N, N);
            dres_dyn.copy_from(&dres);
            let lu = dres_dyn.lu();
            let result = lu.solve(&res);
            match result {
                Some(sol) => sol_1 = sol_0 - sol,
                None => {
                    let mut dres_dyn = DMatrix::<f64>::zeros(N, N);
                    dres_dyn.copy_from(&dres);
                    let svd = &dres_dyn.svd_unordered(false, false);

                    return Err(SolverError {
                        sigma_error_res: res_sigma_norm,
                        sigma_error_incr: sigma_error_incr,
                        kappa_error_res: res_kappa_norm,
                        kappa_error_incr: kappa_error_incr,
                        f_error_res: res_f_norm,
                        lambda_error_incr: lambda_error_incr,
                        singular_values: svd.singular_values.as_slice().to_vec(),
                        condition: svd.singular_values.max() / svd.singular_values.min(),
                    });
                }
            }

            // extract solution and calcualte new residual
            sigma = sol_1.fixed_rows::<STRESS_STRAIN>(0).into();
            kappa = sol_1.fixed_rows::<KAPPA>(STRESS_STRAIN + 1).into();
            del_lambda = sol_1[STRESS_STRAIN];

            sigma_prev = sol_0.fixed_rows::<STRESS_STRAIN>(0).into();
            kappa_prev = sol_0.fixed_rows::<KAPPA>(STRESS_STRAIN + 1).into();
            del_lambda_prev = sol_0[STRESS_STRAIN];

            //Set all states in order to evaluate the new residuals
            self.model.set_model_state(&sigma, &kappa);

            res_sigma = (&sigma - sigma_tr
                + del_lambda * self.model.elastic_tangent() * self.model.g())
                * scaling_factors.0;
            res_kappa = (&kappa - kappa_0 - del_lambda * self.model.k()) * scaling_factors.2;
            res_f = self.model.f() * scaling_factors.1;

            res.fixed_rows_mut::<STRESS_STRAIN>(0).copy_from(&res_sigma);
            res.fixed_rows_mut::<1>(STRESS_STRAIN).x = res_f;
            res.fixed_rows_mut::<KAPPA>(STRESS_STRAIN + 1)
                .copy_from(&res_kappa);

            res_sigma_norm.push(res_sigma.norm());
            res_kappa_norm.push(res_kappa.norm());
            res_f_norm.push(res_f.abs());

            sigma_error_incr.push((sigma - sigma_prev).norm());
            kappa_error_incr.push((kappa - kappa_prev).norm());
            lambda_error_incr.push((del_lambda - del_lambda_prev).abs());
            let converged_res: bool = res_sigma.norm() < self.atol
                && res_kappa[0].abs() < self.atol
                && res_f.abs() < self.atol;
            let converged_incr: bool = (sigma - sigma_prev).norm()
                < self.atol + self.rtol * sigma.norm()
                && (kappa - kappa_prev).norm() < self.atol + self.rtol * kappa.norm()
                && (del_lambda - del_lambda_prev).abs() < self.atol + self.rtol * del_lambda.abs();
            if converged_res {
                break;
            }
            if converged_incr {
                break;
            }
            if i > self.maxit {
                let mut dres_dyn = DMatrix::<f64>::zeros(N, N);
                dres_dyn.copy_from(&dres);
                println!(
                    "dres_dyn: stress row {}",
                    dres_dyn.view((0, 0), (6, 8)).norm()
                );
                println!("dres_dyn: f row {}", dres_dyn.view((6, 0), (1, 8)).norm());
                println!(
                    "dres_dyn: kappa row {}",
                    dres_dyn.view((7, 0), (1, 8)).norm()
                );
                let svd = &dres_dyn.svd_unordered(false, false);
                return Err(SolverError {
                    sigma_error_res: res_sigma_norm,
                    sigma_error_incr: sigma_error_incr,
                    kappa_error_res: res_kappa_norm,
                    kappa_error_incr: kappa_error_incr,
                    f_error_res: res_f_norm,
                    lambda_error_incr: lambda_error_incr,
                    singular_values: svd.singular_values.as_slice().to_vec(),
                    condition: svd.singular_values.max() / svd.singular_values.min(),
                });
            }
            i += 1;
        }
        return Ok(SolverResult {
            sigma,
            kappa,
            del_lambda,
            iterations: i,
        });
    }

    pub fn update_newton_matrix(
        &self,
        dres: &mut SMatrix<f64, N, N>,
        del_lambda: f64,
        scaling_factors: (f64, f64, f64),
    ) {
        assert!(dres.ncols() == dres.nrows() && dres.ncols() == STRESS_STRAIN + KAPPA + 1);
        // fill dres_sigma_dsigma
        dres.fixed_view_mut::<STRESS_STRAIN, STRESS_STRAIN>(0, 0)
            .copy_from(
                &(SMatrix::<f64, STRESS_STRAIN, STRESS_STRAIN>::identity()
                    + self.model.elastic_tangent() * del_lambda * self.model.dg_dsigma()),
            );
        //let mut dres_sigma_dlambda = dres.fixed_view_mut::<6, 1>(0, 7);
        dres.fixed_view_mut::<STRESS_STRAIN, 1>(0, STRESS_STRAIN)
            .copy_from(&(self.model.elastic_tangent() * self.model.g()));
        //let mut dres_sigma_dkappa = dres.fixed_view_mut::<6, 1>(0, 6);
        dres.fixed_view_mut::<STRESS_STRAIN, KAPPA>(0, STRESS_STRAIN + 1)
            .copy_from(&(self.model.elastic_tangent() * del_lambda * self.model.dg_dkappa()));

        //let mut dres_f_dsigma = dres.fixed_view_mut::<1, 6>(7, 0);
        dres.fixed_view_mut::<1, STRESS_STRAIN>(STRESS_STRAIN, 0)
            .copy_from(self.model.df_dsigma());
        //let mut dres_f_dlambda = dres.fixed_view_mut::<1, 1>(7, 7);
        dres.fixed_view_mut::<1, 1>(STRESS_STRAIN, STRESS_STRAIN)
            .copy_from_slice(&[0.0]);
        //let mut dres_f_dkappa = dres.fixed_view_mut::<1, 1>(7, 6);
        dres.fixed_view_mut::<1, KAPPA>(STRESS_STRAIN, STRESS_STRAIN + 1)
            .copy_from(self.model.df_dkappa());

        //let mut dres_kappa_dsigma = dres.fixed_view_mut::<1, 6>(6, 0);
        dres.fixed_view_mut::<KAPPA, STRESS_STRAIN>(STRESS_STRAIN + 1, 0)
            .copy_from(&((-del_lambda) * self.model.dk_dsigma()));
        //let mut dres_kappa_dlambda = dres.fixed_view_mut::<1, 1>(6, 7);
        dres.fixed_view_mut::<KAPPA, 1>(STRESS_STRAIN + 1, STRESS_STRAIN)
            .copy_from(&(-self.model.k()));
        //let mut dres_kappa_dkappa = dres.fixed_view_mut::<1, 1>(6, 6);
        dres.fixed_view_mut::<KAPPA, KAPPA>(STRESS_STRAIN + 1, STRESS_STRAIN + 1)
            .copy_from(
                &(&SMatrix::<f64, KAPPA, KAPPA>::identity() - del_lambda * self.model.dk_dkappa()),
            );
        dres.fixed_view_mut::<STRESS_STRAIN, N>(0, 0)
            .scale_mut(scaling_factors.0);
        dres.fixed_view_mut::<1, N>(STRESS_STRAIN, 0)
            .scale_mut(scaling_factors.1);
        dres.fixed_view_mut::<KAPPA, N>(STRESS_STRAIN + 1, 0)
            .scale_mut(scaling_factors.2);
    }
}

impl<
    'a,
    const N: usize,
    const STRESS_STRAIN: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    const KAPPA: usize,
    const KAPPA_P1: usize, //TODO: this is not needed if const generics allows arithmetics
    T: Plasticity<STRESS_STRAIN, N_PARAMETERS, PARAMETERS, KAPPA>,
> NewtonSchurComplementSolver<'a, N, STRESS_STRAIN, N_PARAMETERS, PARAMETERS, KAPPA, KAPPA_P1, T>
{
    pub fn new(model: &'a mut T, atol: f64, rtol: f64, maxit: usize) -> Self {
        assert_eq!(N, STRESS_STRAIN + 1 + KAPPA);
        Self {
            atol,
            rtol,
            maxit,
            model,
        }
    }

    pub fn solve(
        &mut self,
        sigma_tr: &SVector<f64, STRESS_STRAIN>,
        kappa_0: &SVector<f64, KAPPA>,
    ) -> Result<SolverResult<STRESS_STRAIN, KAPPA>, SchurComplementSolverError> {
        let mut del_lambda = 0.0;
        let mut sigma = *sigma_tr;
        let mut kappa = *kappa_0;

        let mut sol_rest = SVector::<f64, KAPPA_P1>::zeros();
        let mut sol_rest_prev = SVector::<f64, KAPPA_P1>::zeros();
        sol_rest.fixed_rows_mut::<KAPPA>(1).copy_from(kappa_0);

        // sol_1.fixed_rows_mut::<STRESS_STRAIN>(0).copy_from(sigma_tr);
        // sol_1.fixed_rows_mut::<1>(STRESS_STRAIN).x = 0.0;
        // sol_1
        //     .fixed_rows_mut::<KAPPA>(STRESS_STRAIN + 1)
        //     .copy_from(kappa_0);

        let (s_sig, s_f, s_kappa) = self.model.residual_scaling(sigma_tr, kappa_0);

        let mut res_sigma = SVector::<f64, STRESS_STRAIN>::zeros();
        let mut res_kappa = SVector::<f64, KAPPA>::zeros();
        let mut res_f = self.model.f();

        let mut res_rest = SVector::<f64, KAPPA_P1>::zeros();
        res_rest[0] = self.model.f();

        //let mut dres = SMatrix::<f64, N, N>::zeros();

        let mut Jss = SMatrix::<f64, STRESS_STRAIN, STRESS_STRAIN>::zeros();
        let mut Jsr = SMatrix::<f64, STRESS_STRAIN, KAPPA_P1>::zeros();
        let mut Jrs = SMatrix::<f64, KAPPA_P1, STRESS_STRAIN>::zeros();
        let mut Jrr = SMatrix::<f64, KAPPA_P1, KAPPA_P1>::zeros();
        let mut Jss_inv: SMatrix<f64, STRESS_STRAIN, STRESS_STRAIN>;
        let mut schur_complement_inv: SMatrix<f64, KAPPA_P1, KAPPA_P1>;
        //let mut res = SVector::<f64, N>::zeros();
        //res[STRESS_STRAIN] = res_f;

        let mut i = 0;

        let mut sigma_prev: SVector<f64, STRESS_STRAIN>;
        let mut kappa_prev: SVector<f64, KAPPA>;
        let mut del_lambda_prev: f64;
        let mut res_sigma_norm: Vec<f64> = vec![res_sigma.norm()];
        let mut res_kappa_norm: Vec<f64> = vec![res_kappa.norm()];
        let mut res_f_norm: Vec<f64> = vec![res_f];
        let mut sigma_error_incr: Vec<f64> = vec![0.0];
        let mut kappa_error_incr: Vec<f64> = vec![0.0];
        let mut lambda_error_incr: Vec<f64> = vec![0.0];
        loop {
            sigma_prev = sigma;
            kappa_prev = kappa;
            del_lambda_prev = del_lambda;

            //Fill the newton matrix
            self.update_matrices(&mut Jss, &mut Jsr, &mut Jrs, &mut Jrr, del_lambda);

            let Jss_inv_res = Jss.try_inverse(); //TODO

            match Jss_inv_res {
                None => {
                    return Err(SchurComplementSolverError {
                        sigma_error_res: res_sigma_norm,
                        sigma_error_incr: sigma_error_incr,
                        kappa_error_res: res_kappa_norm,
                        kappa_error_incr: kappa_error_incr,
                        f_error_res: res_f_norm,
                        lambda_error_incr: lambda_error_incr,
                        //condition_jss: f64::INFINITY,
                        //condition_schur_complement: 0.0,
                    });
                }
                Some(Jss_inv_) => {
                    Jss_inv = Jss_inv_;
                }
            };
            let schur_complement_inv_res = (&Jrr - &(&Jrs * &Jss_inv * &Jsr)).try_inverse();

            match schur_complement_inv_res {
                None => {
                    //let mut Jss_dyn = DMatrix::<f64>::zeros(STRESS_STRAIN, STRESS_STRAIN);
                    //Jss_dyn.copy_from(&Jss);
                    //let svd = Jss_dyn.svd_unordered(false, false);

                    return Err(SchurComplementSolverError {
                        sigma_error_res: res_sigma_norm,
                        sigma_error_incr: sigma_error_incr,
                        kappa_error_res: res_kappa_norm,
                        kappa_error_incr: kappa_error_incr,
                        f_error_res: res_f_norm,
                        lambda_error_incr: lambda_error_incr,
                        //condition_jss: svd.singular_values.max() / svd.singular_values.min(),
                        //condition_schur_complement: f64::INFINITY,
                    });
                }
                Some(schur_inv_) => schur_complement_inv = schur_inv_,
            };

            let del_rest = &schur_complement_inv * (&res_rest - &Jrs * &Jss_inv * &res_sigma);
            let del_sigma = &Jss_inv * &(&res_sigma - &Jsr * &del_rest);
            //line search
            //let mut alpha = 1.0;
            //let mut del_lambda_trial = 0.0;
            // let mut kappa_trial = SVector::<f64, KAPPA>::zeros();
            // let mut sigma_trial = SVector::<f64, STRESS_STRAIN>::zeros();
            // let res_f_current = self.model.f();

            // while del_lambda_trial < del_lambda {
            //     del_lambda_trial = del_lambda - alpha * del_rest[0];

            //     if del_lambda_trial < 0.0 {
            //         alpha *= 0.5;
            //         continue;
            //     }
            //     //rest_trial = &sol_rest - del_rest;
            //     kappa_trial = kappa - alpha * del_rest.fixed_rows(1);
            //     sigma_trial = sigma - alpha * &del_sigma;
            //     //sigma_trial = sigma_tr -del_lambda_trial * self.model.elastic_tangent()*self.model.g();
            //     let f_trial = self.model.calculate_f_only(&sigma_trial, &kappa_trial);

            //     if f_trial.abs() < res_f_current.abs() {
            //         break;
            //     }

            //     alpha *= 0.5
            // }



            //sol_rest -= alpha * &del_rest;
            //sigma -= &Jss_inv * &(&res_sigma - &Jsr * &del_rest);
            //sigma = sigma_trial;
            sol_rest -= &del_rest;
            sigma -= &del_sigma;
            kappa = sol_rest.fixed_rows::<KAPPA>(1).into();
            del_lambda = sol_rest[0];
            //assert!(del_lambda>0.0);

            //Set all states in order to evaluate the new residuals
            self.model.set_model_state(&sigma, &kappa);

            res_sigma =
                &sigma - sigma_tr + del_lambda * self.model.elastic_tangent() * self.model.g();
            res_kappa = &kappa - kappa_0 - del_lambda * self.model.k();
            res_f = self.model.f();

            //println!(
            //    "iteration {}, sigma {}, kappa {}, del_lambda {}, f {}",
            //    i, sigma, kappa, del_lambda, res_f
            //);

            res_rest.fixed_rows_mut::<1>(0).x = res_f;
            res_rest.fixed_rows_mut::<KAPPA>(1).copy_from(&res_kappa);

            res_sigma_norm.push(res_sigma.norm()*s_sig);
            res_kappa_norm.push(res_kappa.norm()*s_kappa);
            res_f_norm.push(res_f.abs()*s_f);

            sigma_error_incr.push((sigma - sigma_prev).norm()*s_sig);
            kappa_error_incr.push((kappa - kappa_prev).norm()*s_kappa);
            lambda_error_incr.push((del_lambda - del_lambda_prev).abs());
            let converged_res: bool = res_sigma.norm() < self.atol
                && res_kappa[0].abs() < self.atol
                && res_f.abs() < self.atol;
            let converged_incr: bool = (sigma - sigma_prev).norm()
                < self.atol + self.rtol * sigma.norm()
                && (kappa - kappa_prev).norm() < self.atol + self.rtol * kappa.norm()
                && (del_lambda - del_lambda_prev).abs() < self.atol + self.rtol * del_lambda.abs();
            if converged_res {
                break;
            }
            if converged_incr {
                break;
            }
            if i > self.maxit {
                //let mut Jss_dyn = DMatrix::<f64>::zeros(STRESS_STRAIN, STRESS_STRAIN);
                //Jss_dyn.copy_from(&Jss);
                //let svd = Jss_dyn.svd_unordered(false, false);
                //let mut schur_complement_inv_dyn = DMatrix::<f64>::zeros(KAPPA_P1, KAPPA_P1);
                //schur_complement_inv_dyn.copy_from(&schur_complement_inv);
                //let schur_svd = schur_complement_inv_dyn.svd_unordered(false, false);
                return Err(SchurComplementSolverError {
                    sigma_error_res: res_sigma_norm,
                    sigma_error_incr: sigma_error_incr,
                    kappa_error_res: res_kappa_norm,
                    kappa_error_incr: kappa_error_incr,
                    f_error_res: res_f_norm,
                    lambda_error_incr: lambda_error_incr,
                    //condition_jss: svd.singular_values.max() / svd.singular_values.min(),
                    //condition_schur_complement: schur_svd.singular_values.max()
                    //    / schur_svd.singular_values.min(),
                });
            }
            i += 1;
        }
        return Ok(SolverResult {
            sigma,
            kappa,
            del_lambda,
            iterations: i,
        });
    }
    pub fn update_matrices(
        &self,
        Jss: &mut SMatrix<f64, STRESS_STRAIN, STRESS_STRAIN>,
        Jsr: &mut SMatrix<f64, STRESS_STRAIN, KAPPA_P1>,
        Jrs: &mut SMatrix<f64, KAPPA_P1, STRESS_STRAIN>,
        Jrr: &mut SMatrix<f64, KAPPA_P1, KAPPA_P1>,
        del_lambda: f64,
    ) {
        Jss.copy_from(
            &(SMatrix::<f64, STRESS_STRAIN, STRESS_STRAIN>::identity()
                + self.model.elastic_tangent() * del_lambda * self.model.dg_dsigma()),
        );
        Jsr.fixed_view_mut::<STRESS_STRAIN, 1>(0, 0)
            .copy_from(&(self.model.elastic_tangent() * self.model.g()));
        Jsr.fixed_view_mut::<STRESS_STRAIN, KAPPA>(0, 1)
            .copy_from(&(self.model.elastic_tangent() * del_lambda * self.model.dg_dkappa()));

        Jrs.fixed_view_mut::<1, STRESS_STRAIN>(0, 0)
            .copy_from(self.model.df_dsigma());
        Jrs.fixed_view_mut::<KAPPA, STRESS_STRAIN>(1, 0)
            .copy_from(&((-del_lambda) * self.model.dk_dsigma()));

        Jrr[(0, 0)] = 0.0;
        Jrr.fixed_view_mut::<1, KAPPA>(0, 1)
            .copy_from(self.model.df_dkappa());
        Jrr.fixed_view_mut::<KAPPA, 1>(1, 0)
            .copy_from(&(-self.model.k()));
        Jrr.fixed_view_mut::<KAPPA, KAPPA>(1, 1).copy_from(
            &(&SMatrix::<f64, KAPPA, KAPPA>::identity() - del_lambda * self.model.dk_dkappa()),
        );
    }
    pub fn update_newton_matrix(
        &self,
        dres: &mut SMatrix<f64, N, N>,
        del_lambda: f64,
        scaling_factors: (f64, f64, f64),
    ) {
        assert!(dres.ncols() == dres.nrows() && dres.ncols() == STRESS_STRAIN + KAPPA + 1);
        // fill dres_sigma_dsigma
        dres.fixed_view_mut::<STRESS_STRAIN, STRESS_STRAIN>(0, 0)
            .copy_from(
                &(SMatrix::<f64, STRESS_STRAIN, STRESS_STRAIN>::identity()
                    + self.model.elastic_tangent() * del_lambda * self.model.dg_dsigma()),
            );
        //let mut dres_sigma_dlambda = dres.fixed_view_mut::<6, 1>(0, 7);
        dres.fixed_view_mut::<STRESS_STRAIN, 1>(0, STRESS_STRAIN)
            .copy_from(&(self.model.elastic_tangent() * self.model.g()));
        //let mut dres_sigma_dkappa = dres.fixed_view_mut::<6, 1>(0, 6);
        dres.fixed_view_mut::<STRESS_STRAIN, KAPPA>(0, STRESS_STRAIN + 1)
            .copy_from(&(self.model.elastic_tangent() * del_lambda * self.model.dg_dkappa()));

        //let mut dres_f_dsigma = dres.fixed_view_mut::<1, 6>(7, 0);
        dres.fixed_view_mut::<1, STRESS_STRAIN>(STRESS_STRAIN, 0)
            .copy_from(self.model.df_dsigma());
        //let mut dres_f_dlambda = dres.fixed_view_mut::<1, 1>(7, 7);
        dres.fixed_view_mut::<1, 1>(STRESS_STRAIN, STRESS_STRAIN)
            .copy_from_slice(&[0.0]);
        //let mut dres_f_dkappa = dres.fixed_view_mut::<1, 1>(7, 6);
        dres.fixed_view_mut::<1, KAPPA>(STRESS_STRAIN, STRESS_STRAIN + 1)
            .copy_from(self.model.df_dkappa());

        //let mut dres_kappa_dsigma = dres.fixed_view_mut::<1, 6>(6, 0);
        dres.fixed_view_mut::<KAPPA, STRESS_STRAIN>(STRESS_STRAIN + 1, 0)
            .copy_from(&((-del_lambda) * self.model.dk_dsigma()));
        //let mut dres_kappa_dlambda = dres.fixed_view_mut::<1, 1>(6, 7);
        dres.fixed_view_mut::<KAPPA, 1>(STRESS_STRAIN + 1, STRESS_STRAIN)
            .copy_from(&(-self.model.k()));
        //let mut dres_kappa_dkappa = dres.fixed_view_mut::<1, 1>(6, 6);
        dres.fixed_view_mut::<KAPPA, KAPPA>(STRESS_STRAIN + 1, STRESS_STRAIN + 1)
            .copy_from(
                &(&SMatrix::<f64, KAPPA, KAPPA>::identity() - del_lambda * self.model.dk_dkappa()),
            );
        dres.fixed_view_mut::<STRESS_STRAIN, N>(0, 0)
            .scale_mut(scaling_factors.0);
        dres.fixed_view_mut::<1, N>(STRESS_STRAIN, 0)
            .scale_mut(scaling_factors.1);
        dres.fixed_view_mut::<KAPPA, N>(STRESS_STRAIN + 1, 0)
            .scale_mut(scaling_factors.2);
    }
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
{
    pub fn new(model: &'a mut T, atol: f64, rtol: f64, maxit: usize) -> Self {
        assert_eq!(N, STRESS_STRAIN + 1 + KAPPA);
        Self {
            atol,
            rtol,
            maxit,
            model,
        }
    }

    pub fn solve(
        &mut self,
        sigma_tr: &SVector<f64, STRESS_STRAIN>,
        kappa_0: &SVector<f64, KAPPA>,
    ) -> Result<SolverResult<STRESS_STRAIN, KAPPA>, SolverError> {
        let mut del_lambda = 0.0;
        let mut sigma: SVector<f64, STRESS_STRAIN>;
        let mut del_eps_pl: SVector<f64,STRESS_STRAIN>;
        let mut kappa: SVector<f64, KAPPA>;
        let mut sol_0 = SVector::<f64, N>::zeros();
        let mut sol_1 = SVector::<f64, N>::zeros();
        //sol_1.fixed_rows_mut::<STRESS_STRAIN>(0).copy_from(sigma_tr);
        sol_1.fixed_rows_mut::<1>(STRESS_STRAIN).x = 0.0;
        sol_1
            .fixed_rows_mut::<KAPPA>(STRESS_STRAIN + 1)
            .copy_from(kappa_0);

        let scaling_factors = self.model.residual_scaling(sigma_tr, kappa_0);

        let mut res_eps = SVector::<f64, STRESS_STRAIN>::zeros();
        let mut res_kappa = SVector::<f64, KAPPA>::zeros();
        let mut res_f = self.model.f() * scaling_factors.1;

        let mut dres = SMatrix::<f64, N, N>::zeros();

        let mut res = SVector::<f64, N>::zeros();
        res[STRESS_STRAIN] = res_f;

        let mut i = 0;

        let mut del_eps_pl_prev: SVector<f64, STRESS_STRAIN>;
        let mut kappa_prev: SVector<f64, KAPPA>;
        let mut del_lambda_prev: f64;
        let mut res_eps_norm: Vec<f64> = vec![res_eps.norm()];
        let mut res_kappa_norm: Vec<f64> = vec![res_kappa.norm()];
        let mut res_f_norm: Vec<f64> = vec![res_f];
        let mut eps_error_incr: Vec<f64> = vec![0.0];
        let mut kappa_error_incr: Vec<f64> = vec![0.0];
        let mut lambda_error_incr: Vec<f64> = vec![0.0];
        loop {
            sol_0 = sol_1;

            //Fill the newton matrix
            self.update_strain_based_matrix(&mut dres, del_lambda, scaling_factors);

            //this part of the code is ugly
            let mut dres_dyn = DMatrix::<f64>::zeros(N, N);
            dres_dyn.copy_from(&dres);
            let lu = dres_dyn.lu();
            let result = lu.solve(&res);
            match result {
                Some(sol) => sol_1 = sol_0 - sol,
                None => {
                    let mut dres_dyn = DMatrix::<f64>::zeros(N, N);
                    dres_dyn.copy_from(&dres);
                    let svd = &dres_dyn.svd_unordered(false, false);

                    return Err(SolverError {
                        sigma_error_res: res_eps_norm,
                        sigma_error_incr: eps_error_incr,
                        kappa_error_res: res_kappa_norm,
                        kappa_error_incr: kappa_error_incr,
                        f_error_res: res_f_norm,
                        lambda_error_incr: lambda_error_incr,
                        singular_values: svd.singular_values.as_slice().to_vec(),
                        condition: svd.singular_values.max() / svd.singular_values.min(),
                    });
                }
            }

            // extract solution and calcualte new residual
            del_eps_pl = sol_1.fixed_rows::<STRESS_STRAIN>(0).into();
            kappa = sol_1.fixed_rows::<KAPPA>(STRESS_STRAIN + 1).into();
            del_lambda = sol_1[STRESS_STRAIN];

            del_eps_pl_prev = sol_0.fixed_rows::<STRESS_STRAIN>(0).into();
            kappa_prev = sol_0.fixed_rows::<KAPPA>(STRESS_STRAIN + 1).into();
            del_lambda_prev = sol_0[STRESS_STRAIN];


            sigma = sigma_tr - self.model.elastic_tangent()*&del_eps_pl;
            //Set all states in order to evaluate the new residuals
            self.model.set_model_state(&sigma, &kappa);

            res_eps = del_eps_pl - del_lambda * self.model.g();
                //+ del_lambda * self.model.elastic_tangent() * self.model.g())
                //* scaling_factors.0;
            res_kappa = (&kappa - kappa_0 - del_lambda * self.model.k());// * scaling_factors.2;
            res_f = self.model.f()* scaling_factors.1;

            res.fixed_rows_mut::<STRESS_STRAIN>(0).copy_from(&res_eps);
            res.fixed_rows_mut::<1>(STRESS_STRAIN).x = res_f;
            res.fixed_rows_mut::<KAPPA>(STRESS_STRAIN + 1)
                .copy_from(&res_kappa);

            res_eps_norm.push(res_eps.norm());
            res_kappa_norm.push(res_kappa.norm());
            res_f_norm.push(res_f.abs());

            eps_error_incr.push((del_eps_pl - del_eps_pl_prev).norm());
            kappa_error_incr.push((kappa - kappa_prev).norm());
            lambda_error_incr.push((del_lambda - del_lambda_prev).abs());
            let converged_res: bool = res_eps.norm() < self.atol
                && res_kappa[0].abs() < self.atol
                && res_f.abs() < self.atol;
            let converged_incr: bool = (del_eps_pl -del_eps_pl_prev).norm()
                < self.atol + self.rtol * del_eps_pl.norm()
                && (kappa - kappa_prev).norm() < self.atol + self.rtol * kappa.norm()
                && (del_lambda - del_lambda_prev).abs() < self.atol + self.rtol * del_lambda.abs();
            if converged_res {
                break;
            }
            if converged_incr {
                break;
            }
            if i > self.maxit {
                let mut dres_dyn = DMatrix::<f64>::zeros(N, N);
                dres_dyn.copy_from(&dres);
                println!(
                    "dres_dyn: stress row {}",
                    dres_dyn.view((0, 0), (6, 8)).norm()
                );
                println!("dres_dyn: f row {}", dres_dyn.view((6, 0), (1, 8)).norm());
                println!(
                    "dres_dyn: kappa row {}",
                    dres_dyn.view((7, 0), (1, 8)).norm()
                );
                let svd = &dres_dyn.svd_unordered(false, false);
                return Err(SolverError {
                    sigma_error_res: res_eps_norm,
                    sigma_error_incr: eps_error_incr,
                    kappa_error_res: res_kappa_norm,
                    kappa_error_incr: kappa_error_incr,
                    f_error_res: res_f_norm,
                    lambda_error_incr: lambda_error_incr,
                    singular_values: svd.singular_values.as_slice().to_vec(),
                    condition: svd.singular_values.max() / svd.singular_values.min(),
                });
            }
            i += 1;
        }
        return Ok(SolverResult {
            sigma,
            kappa,
            del_lambda,
            iterations: i,
        });
    }

    pub fn update_newton_matrix(
        &self,
        dres: &mut SMatrix<f64, N, N>,
        del_lambda: f64,
        scaling_factors: (f64, f64, f64),
    ) {
        assert!(dres.ncols() == dres.nrows() && dres.ncols() == STRESS_STRAIN + KAPPA + 1);
        // fill dres_sigma_dsigma
        dres.fixed_view_mut::<STRESS_STRAIN, STRESS_STRAIN>(0, 0)
            .copy_from(
                &(SMatrix::<f64, STRESS_STRAIN, STRESS_STRAIN>::identity()
                    + self.model.elastic_tangent() * del_lambda * self.model.dg_dsigma()),
            );
        //let mut dres_sigma_dlambda = dres.fixed_view_mut::<6, 1>(0, 7);
        dres.fixed_view_mut::<STRESS_STRAIN, 1>(0, STRESS_STRAIN)
            .copy_from(&(self.model.elastic_tangent() * self.model.g()));
        //let mut dres_sigma_dkappa = dres.fixed_view_mut::<6, 1>(0, 6);
        dres.fixed_view_mut::<STRESS_STRAIN, KAPPA>(0, STRESS_STRAIN + 1)
            .copy_from(&(self.model.elastic_tangent() * del_lambda * self.model.dg_dkappa()));

        //let mut dres_f_dsigma = dres.fixed_view_mut::<1, 6>(7, 0);
        dres.fixed_view_mut::<1, STRESS_STRAIN>(STRESS_STRAIN, 0)
            .copy_from(self.model.df_dsigma());
        //let mut dres_f_dlambda = dres.fixed_view_mut::<1, 1>(7, 7);
        dres.fixed_view_mut::<1, 1>(STRESS_STRAIN, STRESS_STRAIN)
            .copy_from_slice(&[0.0]);
        //let mut dres_f_dkappa = dres.fixed_view_mut::<1, 1>(7, 6);
        dres.fixed_view_mut::<1, KAPPA>(STRESS_STRAIN, STRESS_STRAIN + 1)
            .copy_from(self.model.df_dkappa());

        //let mut dres_kappa_dsigma = dres.fixed_view_mut::<1, 6>(6, 0);
        dres.fixed_view_mut::<KAPPA, STRESS_STRAIN>(STRESS_STRAIN + 1, 0)
            .copy_from(&((-del_lambda) * self.model.dk_dsigma()));
        //let mut dres_kappa_dlambda = dres.fixed_view_mut::<1, 1>(6, 7);
        dres.fixed_view_mut::<KAPPA, 1>(STRESS_STRAIN + 1, STRESS_STRAIN)
            .copy_from(&(-self.model.k()));
        //let mut dres_kappa_dkappa = dres.fixed_view_mut::<1, 1>(6, 6);
        dres.fixed_view_mut::<KAPPA, KAPPA>(STRESS_STRAIN + 1, STRESS_STRAIN + 1)
            .copy_from(
                &(&SMatrix::<f64, KAPPA, KAPPA>::identity() - del_lambda * self.model.dk_dkappa()),
            );
        dres.fixed_view_mut::<STRESS_STRAIN, N>(0, 0)
            .scale_mut(scaling_factors.0);
        dres.fixed_view_mut::<1, N>(STRESS_STRAIN, 0)
            .scale_mut(scaling_factors.1);
        dres.fixed_view_mut::<KAPPA, N>(STRESS_STRAIN + 1, 0)
            .scale_mut(scaling_factors.2);
    }
    
    pub fn update_strain_based_matrix(
        &self,
        dres: &mut SMatrix<f64, N, N>,
        del_lambda: f64,
        scaling_factors: (f64, f64, f64),
    ) {
        assert!(dres.ncols() == dres.nrows() && dres.ncols() == STRESS_STRAIN + KAPPA + 1);
        // fill dres_sigma_dsigma
        dres.fixed_view_mut::<STRESS_STRAIN, STRESS_STRAIN>(0, 0)
            .copy_from(
                &(SMatrix::<f64, STRESS_STRAIN, STRESS_STRAIN>::identity()
                    + del_lambda * self.model.dg_dsigma()*self.model.elastic_tangent()),
            );
        //let mut dres_sigma_dlambda = dres.fixed_view_mut::<6, 1>(0, 7);
        dres.fixed_view_mut::<STRESS_STRAIN, 1>(0, STRESS_STRAIN)
            .copy_from(&( -self.model.g()));
        //let mut dres_sigma_dkappa = dres.fixed_view_mut::<6, 1>(0, 6);
        dres.fixed_view_mut::<STRESS_STRAIN, KAPPA>(0, STRESS_STRAIN + 1)
            .copy_from(&(- del_lambda * self.model.dg_dkappa()));

        //let mut dres_f_dsigma = dres.fixed_view_mut::<1, 6>(7, 0);
        dres.fixed_view_mut::<1, STRESS_STRAIN>(STRESS_STRAIN, 0)
            .copy_from(&(-self.model.df_dsigma()*self.model.elastic_tangent()));
        //let mut dres_f_dlambda = dres.fixed_view_mut::<1, 1>(7, 7);
        dres.fixed_view_mut::<1, 1>(STRESS_STRAIN, STRESS_STRAIN)
            .copy_from_slice(&[0.0]);
        //let mut dres_f_dkappa = dres.fixed_view_mut::<1, 1>(7, 6);
        dres.fixed_view_mut::<1, KAPPA>(STRESS_STRAIN, STRESS_STRAIN + 1)
            .copy_from(self.model.df_dkappa());

        //let mut dres_kappa_dsigma = dres.fixed_view_mut::<1, 6>(6, 0);
        dres.fixed_view_mut::<KAPPA, STRESS_STRAIN>(STRESS_STRAIN + 1, 0)
            .copy_from(&(del_lambda * self.model.dk_dsigma()*self.model.elastic_tangent()));
        //let mut dres_kappa_dlambda = dres.fixed_view_mut::<1, 1>(6, 7);
        dres.fixed_view_mut::<KAPPA, 1>(STRESS_STRAIN + 1, STRESS_STRAIN)
            .copy_from(&(-self.model.k()));
        //let mut dres_kappa_dkappa = dres.fixed_view_mut::<1, 1>(6, 6);
        dres.fixed_view_mut::<KAPPA, KAPPA>(STRESS_STRAIN + 1, STRESS_STRAIN + 1)
            .copy_from(
                &(&SMatrix::<f64, KAPPA, KAPPA>::identity() - del_lambda * self.model.dk_dkappa()),
            );
        //dres.fixed_view_mut::<STRESS_STRAIN, N>(0, 0)
        //    .scale_mut(scaling_factors.0);
        dres.fixed_view_mut::<1, N>(STRESS_STRAIN, 0)
            .scale_mut(scaling_factors.1);
        //dres.fixed_view_mut::<KAPPA, N>(STRESS_STRAIN + 1, 0)
        //    .scale_mut(scaling_factors.2);
    }
}