use std::marker::PhantomData;

use crate::QDim; // Ensure QDim is imported from the correct module
use crate::general::Plasticity;
use crate::newton_solvers::{NewtonSchurComplementSolver, NewtonSolver};
use crate::plasticity::NewtonSolverStrain;
use crate::{
    create_history_parameter_struct,
    interfaces::{ArrayEquivalent, GradientConstitutiveModelFn, NonlocalTangents, StaticMap},
};
//stuff

use nalgebra::{RowSVector, SMatrix, SVector};
pub trait GradientPlasticity<
    const STRESS_STRAIN: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    const KAPPA: usize,
>: Plasticity<STRESS_STRAIN, N_PARAMETERS, PARAMETERS, KAPPA>
{
    //must be used once
    fn set_nonlocal_state(
        &mut self,
        kappa_nonlocal: &SVector<f64, KAPPA>,
        kappa_nonlocal_max: &SVector<f64, KAPPA>,
    );

    //must only be used when determining the tangents of the model
    fn set_nonlocal_derivatives(
        &mut self,
        sigma: &SVector<f64, STRESS_STRAIN>,
        kappa: &SVector<f64, KAPPA>,
    );
    fn df_dkappa_nonlocal(&self) -> &RowSVector<f64, KAPPA>;
    fn dg_dkappa_nonlocal(&self) -> &SMatrix<f64, STRESS_STRAIN, KAPPA>;
    fn dk_dkappa_nonlocal(&self) -> &SMatrix<f64, KAPPA, KAPPA>;
}

pub struct IsotropicGradientPlasticityModel3D<
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    MODEL: GradientPlasticity<6, N_PARAMETERS, PARAMETERS, 1>,
> {
    phantom: PhantomData<MODEL>,
}

create_history_parameter_struct!(
    IsotropicGradientPlasticityHistory3D,
    2,
    7,
    [
        (alpha_nonlocal_max, (QDim::Scalar)),
        (plastic_strain, (QDim::RotatableVector(6)))
    ]
);

impl<
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    MODEL: GradientPlasticity<6, N_PARAMETERS, PARAMETERS, 1>,
> GradientConstitutiveModelFn<6, 2, 7, N_PARAMETERS, PARAMETERS>
    for IsotropicGradientPlasticityModel3D<N_PARAMETERS, PARAMETERS, MODEL>
{
    type History = IsotropicGradientPlasticityHistory3D;
    type Parameters = MODEL::Parameters;
    #[inline]
    fn evaluate(
        time: f64,
        del_time: f64,
        del_strain: &[f64; 6],
        nonlocal_quantity: &[f64; 1],
        stress: &mut [f64; 6],
        local_quantity: &mut [f64; 1],
        tangents: Option<&mut NonlocalTangents<6>>,
        history: &mut [f64; 7],
        parameters: &[f64; PARAMETERS],
    ) {
        let parameters_ = Self::Parameters::from_array(parameters);
        let history_ = IsotropicGradientPlasticityHistory3D::from_array_mut(history);
        let mut model = MODEL::new(parameters_);

        let del_eps = SVector::<f64, 6>::from_column_slice(del_strain);
        let sigma_0 = SVector::<f64, 6>::from_column_slice(stress);
        let sigma_tr = model.elastic_tangent() * del_eps + sigma_0;

        let alpha_0 = SVector::<f64, 1>::from_column_slice(local_quantity);
        let alpha_nonlocal = SVector::<f64, 1>::from_column_slice(nonlocal_quantity);
        let alpha_nonlocal_max = SVector::<f64, 1>::from_element(f64::max(
            alpha_nonlocal.x,
            history_.alpha_nonlocal_max,
        ));
        let old_max = history_.alpha_nonlocal_max;
        history_.alpha_nonlocal_max = alpha_nonlocal_max.x;

        model.set_nonlocal_state(&alpha_nonlocal, &alpha_nonlocal_max);
        model.set_model_state(&sigma_tr, &alpha_0);

        let f = model.f();
        if f <= 0.0 {
            *stress = sigma_tr.data.0[0];
            if let Some(tangents) = tangents {
                tangents.dsigma_deps = *model.elastic_tangent();
                tangents.dlocal_deps = RowSVector::<f64, 6>::zeros();
                tangents.dsigma_dnonlocal = SVector::<f64, 6>::zeros();
                tangents.dlocal_dnonlocal = 0.0;
            }
            return;
        } else {
            //let mut solver =
            //    NewtonSchurComplementSolver::<8, 6, N_PARAMETERS, PARAMETERS, 1, 2, MODEL>::new(
            //        &mut model, 1e-8, 1e-8, 50,
            //    );
            //let mut solver = NewtonSolver::new(&mut model, 1e-8, 1e-8, 50);
            let mut solver = NewtonSolverStrain::new(&mut model, 1e-8, 1e-8, 50);
            let msg = format!("Plasticity failed to converge on sigma_tr {}, sigma_0 {}, alpha_max_old {}", sigma_tr, sigma_0, old_max);
            let result = solver
                .solve(&sigma_tr, &alpha_0)
                .expect(&msg);

            // Update the stress and history
            *stress = result.sigma.data.0[0];
            *local_quantity = result.kappa.data.0[0];
            history_.plastic_strain += result.del_lambda * solver.model.g();
            if let Some(tangents) = tangents {
                let mut dres = SMatrix::<f64, 8, 8>::zeros();
                solver.update_newton_matrix(&mut dres, result.del_lambda, (1.0, 1.0, 1.0));

                let inverse = dres
                    .try_inverse()
                    .expect("Plasticity3D: Failed to calculate tangent.");

                model.set_nonlocal_derivatives(&result.sigma, &result.kappa);
                let mut nonlocal_derivatives = SMatrix::<f64, 8, 1>::zeros();
                nonlocal_derivatives.fixed_rows_mut::<6>(0).copy_from(
                    &(-result.del_lambda * model.elastic_tangent() * model.dg_dkappa_nonlocal()),
                );
                nonlocal_derivatives
                    .fixed_rows_mut::<1>(6)
                    .copy_from(&(-model.df_dkappa_nonlocal()));
                nonlocal_derivatives
                    .fixed_rows_mut::<1>(7)
                    .copy_from(&(result.del_lambda * model.dk_dkappa_nonlocal()));
                let tangent_vector = &inverse * &nonlocal_derivatives;

                let mut plastic_tangent: SMatrix<f64, 6, 6> =
                    inverse.fixed_view::<6, 6>(0, 0) * model.elastic_tangent();
                plastic_tangent.transpose_mut(); //TODO: move the transpose to the python bindings

                tangents.dsigma_deps = plastic_tangent;
                tangents.dlocal_deps = inverse.fixed_view::<1, 6>(7, 0) * model.elastic_tangent();
                tangents.dsigma_dnonlocal = tangent_vector.fixed_rows::<6>(0).into();
                tangents.dlocal_dnonlocal = tangent_vector[7];
            }
        }
    }
}
