use std::marker::PhantomData;

use crate::QDim; // Ensure QDim is imported from the correct module
use crate::{
    create_history_parameter_struct,
    interfaces::{ArrayEquivalent, ConstitutiveModelFn, StaticMap},
};
use konst::result;
use nalgebra::{DMatrix, RowSVector, SMatrix, SVector};
pub trait Plasticity<
    const STRESS_STRAIN: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    const KAPPA: usize,
>
{
    type Parameters: ArrayEquivalent<PARAMETERS> + StaticMap<N_PARAMETERS, QDim>;

    fn new(parameters: &Self::Parameters) -> Self;
    fn set_model_state(&mut self, sigma: &SVector<f64, STRESS_STRAIN>, kappa: &SVector<f64, KAPPA>);
    fn f(&self) -> f64;
    fn df_dsigma(&self) -> &RowSVector<f64, STRESS_STRAIN>;
    fn df_dkappa(&self) -> &RowSVector<f64, KAPPA>;
    fn g(&self) -> &SVector<f64, STRESS_STRAIN>;
    fn dg_dkappa(&self) -> &SMatrix<f64, STRESS_STRAIN, KAPPA>;
    fn dg_dsigma(&self) -> &SMatrix<f64, STRESS_STRAIN, STRESS_STRAIN>;
    fn k(&self) -> &SVector<f64, KAPPA>;
    fn dk_dsigma(&self) -> &SMatrix<f64, KAPPA, STRESS_STRAIN>;
    fn dk_dkappa(&self) -> &SMatrix<f64, KAPPA, KAPPA>;
    fn elastic_tangent(&self) -> &SMatrix<f64, STRESS_STRAIN, STRESS_STRAIN>;
    fn elastic_tangent_inv(&self) -> &SMatrix<f64, STRESS_STRAIN, STRESS_STRAIN>;
    //fn del_plastic_strain(&self) -> &SVector<f64, STRESS_STRAIN>;.
}

pub struct IsotropicPlasticityModel3D<
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    MODEL: Plasticity<6, N_PARAMETERS, PARAMETERS, 1>,
> {
    phantom: PhantomData<MODEL>,
}

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
pub struct SolverResult<const STRESS_STRAIN: usize, const KAPPA: usize> {
    pub sigma: SVector<f64, STRESS_STRAIN>,
    pub kappa: SVector<f64, KAPPA>,
    pub del_lambda: f64,
    pub iterations: usize,
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
    ) -> Option<SolverResult<STRESS_STRAIN, KAPPA>> {
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

        let mut res_sigma = SVector::<f64, STRESS_STRAIN>::zeros();
        let mut res_kappa = SVector::<f64, KAPPA>::zeros();
        let mut res_f = self.model.f();

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
        loop {
            sol_0 = sol_1;

            //Fill the newton matrix
            self.update_newton_matrix(&mut dres, del_lambda);

            //this part of the code is ugly
            let mut dres_dyn = DMatrix::<f64>::zeros(N, N);
            dres_dyn.copy_from(&dres);
            let lu = dres_dyn.lu();
            let result = lu.solve(&res);
            match result {
                Some(sol) => sol_1 = sol_0 - sol,
                None => return None,
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

            res_sigma =
                &sigma - sigma_tr + del_lambda * self.model.elastic_tangent() * self.model.g();
            res_kappa = &kappa - kappa_0 - del_lambda * self.model.k();
            res_f = self.model.f();

            res.fixed_rows_mut::<STRESS_STRAIN>(0).copy_from(&res_sigma);
            res.fixed_rows_mut::<1>(STRESS_STRAIN).x = res_f;
            res.fixed_rows_mut::<KAPPA>(STRESS_STRAIN + 1)
                .copy_from(&res_kappa);

            res_sigma_norm.push(res_sigma.norm());
            res_kappa_norm.push(res_kappa.norm());
            res_f_norm.push(res_f.abs());

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
                return None;
            }
            i += 1;
        }
        return Some(SolverResult {
            sigma,
            kappa,
            del_lambda,
            iterations: i,
        });
    }

    pub fn update_newton_matrix(&self, dres: &mut SMatrix<f64, N, N>, del_lambda: f64) {
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
    }
}

create_history_parameter_struct!(
    IsotropicPlasticityHistory3D,
    2,
    7,
    [
        (alpha, (QDim::Scalar)),
        (plastic_strain, (QDim::RotatableVector(6)))
    ]
);

impl<
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    MODEL: Plasticity<6, N_PARAMETERS, PARAMETERS, 1>,
> ConstitutiveModelFn<6, 2, 7, N_PARAMETERS, PARAMETERS>
    for IsotropicPlasticityModel3D<N_PARAMETERS, PARAMETERS, MODEL>
{
    type History = IsotropicPlasticityHistory3D;
    type Parameters = MODEL::Parameters;
    #[inline]
    fn evaluate(
        _time: f64,
        _del_time: f64,
        //_strain: &[f64; 6],
        del_strain: &[f64; 6],
        stress: &mut [f64; 6],
        tangent: Option<&mut [[f64; 6]; 6]>,
        history: &mut [f64; 7],
        parameters: &[f64; PARAMETERS],
    ) {
        let parameters_ = Self::Parameters::from_array(parameters);
        let history_ = IsotropicPlasticityHistory3D::from_array_mut(history);
        let mut model = MODEL::new(parameters_);

        let del_eps = SVector::<f64, 6>::from_column_slice(del_strain);
        let sigma_0 = SVector::<f64, 6>::from_column_slice(stress);
        let sigma_tr = model.elastic_tangent() * del_eps + sigma_0;

        let alpha_0 = SVector::<f64, 1>::from_element(history_.alpha);
        //let mut alpha_1 = alpha_0.clone();
        //let mut sigma_1: SVector<f64, 6>;
        model.set_model_state(&sigma_tr, &alpha_0);

        let f = model.f();
        if f <= 0.0 {
            *stress = sigma_tr.data.0[0];
            if let Some(tangent) = tangent {
                *tangent = model.elastic_tangent().data.0;
            }
            return;
        } else {
            let mut solver = NewtonSolver::new(&mut model, 1e-8, 1e-8, 50);

            let result = solver
                .solve(&sigma_tr, &alpha_0)
                .expect("Plasticity model failed to converge");

            // Update the stress and history
            *stress = result.sigma.data.0[0];
            history_.alpha = result.kappa[0];
            history_.plastic_strain += result.del_lambda * solver.model.g();
            if let Some(tangent) = tangent {
                //update the newton matrix in order to evaluate at t_{n+1}
                let mut dres = SMatrix::<f64, 8, 8>::zeros();
                solver.update_newton_matrix(&mut dres, result.del_lambda);
                let inverse = dres
                    .try_inverse()
                    .expect("Plasticity3D: Failed to calculate tangent");
                let mut plastic_tangent: SMatrix<f64, 6, 6> =
                    inverse.fixed_view::<6, 6>(0, 0) * model.elastic_tangent();
                plastic_tangent.transpose_mut(); //TODO: move the transpose to the python bindings
                *tangent = plastic_tangent.data.0;
                
            }
        }
    }
}
