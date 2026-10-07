use std::marker::PhantomData;

use crate::QDim; use crate::plasticity::NewtonSolver;
// Ensure QDim is imported from the correct module
use crate::{
    create_history_parameter_struct,
    interfaces::{ArrayEquivalent, ConstitutiveModelFn, StaticMap},
};
use nalgebra::{RowSVector, SMatrix, SVector};
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
    fn calculate_f_only(&mut self, sigma: &SVector<f64, STRESS_STRAIN>, kappa: &SVector<f64, KAPPA>)->f64;
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
    fn residual_scaling(&self, sigma: &SVector<f64, STRESS_STRAIN>, kappa: &SVector<f64, KAPPA>) -> (f64,f64,f64) {
        return (1.0,1.0,1.0)
    }

    //fn del_plastic_strain(&self) -> &SVector<f64, STRESS_STRAIN>;.
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
                solver.update_newton_matrix(&mut dres, result.del_lambda, (1.0,1.0,1.0));
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
