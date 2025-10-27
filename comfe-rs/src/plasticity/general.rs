use std::marker::PhantomData;

use crate::QDim; // Ensure QDim is imported from the correct module
use crate::{
    create_history_parameter_struct,
    interfaces::{ArrayEquivalent, ConstitutiveModelFn, StaticMap},
};
use nalgebra::{SMatrix, SVector};
pub trait Plasticity<
    const STRESS_STRAIN: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    const KAPPA: usize,
>
{
    type Parameters: ArrayEquivalent<PARAMETERS> + StaticMap<N_PARAMETERS, QDim>;

    fn new(parameters: &Self::Parameters) -> Self;
    fn set_model_state(
        &mut self,
        sigma_0: &SVector<f64, STRESS_STRAIN>,
        sigma_1: &SVector<f64, STRESS_STRAIN>,
        del_eps: &SVector<f64, STRESS_STRAIN>,
        kappa: &SVector<f64, KAPPA>,
    );
    fn f(&self) -> f64;
    fn df_dsigma(&self) -> &SVector<f64, STRESS_STRAIN>;
    fn df_dkappa(&self) -> &SVector<f64, KAPPA>;
    fn g(&self) -> &SVector<f64, STRESS_STRAIN>;
    fn dg_dkappa(&self) -> &SMatrix<f64, STRESS_STRAIN, KAPPA>;
    fn dg_dsigma(&self) -> &SMatrix<f64, STRESS_STRAIN, STRESS_STRAIN>;
    fn k(&self) -> &SVector<f64, KAPPA>;
    fn dk_dsigma(&self) -> &SMatrix<f64, KAPPA, STRESS_STRAIN>;
    fn dk_dkappa(&self) -> &SMatrix<f64, KAPPA, KAPPA>;
    fn elastic_tangent(&self) -> &SMatrix<f64, STRESS_STRAIN, STRESS_STRAIN>;
    fn elastic_tangent_inv(&self) -> &SMatrix<f64, STRESS_STRAIN, STRESS_STRAIN>;
    fn del_plastic_strain(&self) -> &SVector<f64, STRESS_STRAIN>;
    fn update_newton_matrix<const N: usize>(&self, dres: &mut SMatrix<f64, N, N>, del_lambda: f64) {
        assert!(N == STRESS_STRAIN + KAPPA + 1);
        // fill dres_sigma_dsigma
        dres.fixed_view_mut::<STRESS_STRAIN, STRESS_STRAIN>(0, 0)
            .copy_from(
                &(-SMatrix::<f64, STRESS_STRAIN, STRESS_STRAIN>::identity()
                    - self.elastic_tangent() * del_lambda * self.dg_dsigma()),
            );
        //let mut dres_sigma_dkappa = dres.fixed_view_mut::<6, 1>(0, 6);
        dres.fixed_view_mut::<STRESS_STRAIN, KAPPA>(0, STRESS_STRAIN)
            .copy_from(&(-self.elastic_tangent() * del_lambda * self.dg_dkappa()));
        //let mut dres_sigma_dlambda = dres.fixed_view_mut::<6, 1>(0, 7);
        dres.fixed_view_mut::<STRESS_STRAIN, 1>(0, STRESS_STRAIN + 1)
            .copy_from(&(-self.elastic_tangent() * self.g()));

        //let mut dres_kappa_dsigma = dres.fixed_view_mut::<1, 6>(6, 0);
        dres.fixed_view_mut::<KAPPA, STRESS_STRAIN>(STRESS_STRAIN, 0)
            .copy_from(&((-del_lambda)*self.dk_dsigma()));
        //let mut dres_kappa_dkappa = dres.fixed_view_mut::<1, 1>(6, 6);
        dres.fixed_view_mut::<KAPPA, KAPPA>(STRESS_STRAIN, STRESS_STRAIN)
            .copy_from(&(SMatrix::<f64, KAPPA, KAPPA>::identity() - del_lambda * self.dk_dkappa()));
        //let mut dres_kappa_dlambda = dres.fixed_view_mut::<1, 1>(6, 7);
        dres.fixed_view_mut::<KAPPA, 1>(STRESS_STRAIN, STRESS_STRAIN + 1)
            .copy_from(self.k());

        //let mut dres_f_dsigma = dres.fixed_view_mut::<1, 6>(7, 0);
        dres.fixed_view_mut::<1, STRESS_STRAIN>(STRESS_STRAIN + 1, 0)
            .copy_from(&self.df_dsigma().transpose());
        //let mut dres_f_dkappa = dres.fixed_view_mut::<1, 1>(7, 6);
        dres.fixed_view_mut::<1, KAPPA>(STRESS_STRAIN + 1, STRESS_STRAIN)
            .copy_from(&self.df_dkappa().transpose());
        //let mut dres_f_dlambda = dres.fixed_view_mut::<1, 1>(7, 7);
        dres.fixed_view_mut::<1, 1>(STRESS_STRAIN + 1, STRESS_STRAIN + 1)
            .copy_from_slice(&[0.0]);
    }
}

pub struct IsotropicPlasticityModel3D<
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    MODEL: Plasticity<6, N_PARAMETERS, PARAMETERS, 1>,
> {
    phantom: PhantomData<MODEL>,
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
        let mut alpha_1 = alpha_0.clone();
        let mut sigma_1: SVector<f64,6>;
        model.set_model_state(&sigma_0, &sigma_tr, &del_eps, &alpha_0);

        let f = model.f();
        if f <= 0.0 {
            *stress = sigma_tr.data.0[0];
            if let Some(tangent) = tangent {
                *tangent = model.elastic_tangent().data.0;
            }
            return;
        } else {
            let mut del_lambda = 0.0;
            let mut sol_0: SVector<f64, 8>;
            let mut sol_1 = SVector::<f64, 8>::from([
                sigma_tr[0],
                sigma_tr[1],
                sigma_tr[2],
                sigma_tr[3],
                sigma_tr[4],
                sigma_tr[5],
                alpha_0[0],
                0.0,
            ]);
            let mut res_sigma =
                sigma_tr - sigma_0 - del_lambda * model.elastic_tangent() * model.g();
            let mut res_kappa = alpha_1 - alpha_0 - del_lambda * model.k();
            let mut res_f = model.f();

            let mut dres = SMatrix::<f64, 8, 8>::zeros();
            model.update_newton_matrix(&mut dres, 0.0);

            let mut res = SVector::<f64, 8>::from([
                res_sigma[0],
                res_sigma[1],
                res_sigma[2],
                res_sigma[3],
                res_sigma[4],
                res_sigma[5],
                res_kappa[0],
                res_f,
            ]);
            let mut i = 0;
            let maxit = 25;
            let atol = 1e-8;
            let rtol = 1e-8;
            //println!("residual: {}", res);
            let mut sigma_prev: SVector<f64, 6>;
            let mut alpha_prev: SVector<f64, 1>;
            let mut del_lambda_prev: f64;
            loop {
                sol_0 = sol_1;

                let lu = dres.lu();
                let result = lu.solve(&res);
                match result {
                    Some(sol) => sol_1 = sol_0 - sol,
                    None => {
                        panic!(
                            "Plasticity3D: Newton-Raphson failed to solve system {}",
                            dres
                        );
                    }
                }

                // extract solution and calcualte new residual
                sigma_1 = sol_1.fixed_view::<6, 1>(0, 0).into();
                alpha_1 = sol_1.fixed_view::<1, 1>(6, 0).into();
                del_lambda = sol_1[7];

                sigma_prev = sol_0.fixed_view::<6, 1>(0, 0).into();
                alpha_prev = sol_0.fixed_view::<1, 1>(6, 0).into();
                del_lambda_prev = sol_0[7];

                model.set_model_state(&sigma_0, &sigma_1, &del_eps, &alpha_1);
                model.update_newton_matrix(&mut dres, del_lambda);

                res_sigma = &sigma_tr - &sigma_1 - del_lambda * model.elastic_tangent() * model.g();
                res_kappa = &alpha_1 - &alpha_0 - model.k();
                res_f = model.f();

                res = SVector::<f64, 8>::from([
                    res_sigma[0],
                    res_sigma[1],
                    res_sigma[2],
                    res_sigma[3],
                    res_sigma[4],
                    res_sigma[5],
                    res_kappa[0],
                    res_f,
                ]);
                let converged_res: bool =
                    res_sigma.norm() < atol && res_kappa[0].abs() < atol && res_f.abs() < atol;
                let converged_incr: bool = (sigma_1 - sigma_prev).norm()
                    < atol + rtol * sigma_1.norm()
                    && (alpha_1 - alpha_prev)[0].abs() < atol + rtol * alpha_1[0].abs()
                    && (del_lambda - del_lambda_prev).abs() < atol + rtol * del_lambda.abs();
                if converged_res {
                    break;
                }
                if converged_incr {
                    break;
                }
                if i > maxit {
                    panic!(
                        "Plasticity3D: Newton-Raphson did not converge. residual: {}, solution change: {}",
                        res.norm(),
                        (sol_1 - sol_0).norm() / sol_1.norm()
                    );
                }
                i += 1;
            }
            // Update the stress and history
            *stress = sigma_1.data.0[0];
            history_.alpha = alpha_1[0];
            history_.plastic_strain += model.del_plastic_strain();
            if let Some(tangent) = tangent {
                let inverse = dres
                    .try_inverse()
                    .expect("Plasticity3D: Failed to calculate tangent");
                let mut plastic_tangent: SMatrix<f64, 6, 6> =
                    inverse.fixed_view::<6, 6>(0, 0) * 
                    model.elastic_tangent();
                plastic_tangent.transpose_mut();
                *tangent = plastic_tangent.data.0;
            }
        }
    }
}
