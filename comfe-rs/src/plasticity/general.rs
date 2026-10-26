use std::marker::PhantomData;

use crate::QDim;
use crate::plasticity::{NewtonSolverStrain, stress_based_jacobian};
// Ensure QDim is imported from the correct module
use crate::{
    create_history_parameter_struct,
    interfaces::{ArrayEquivalent, ConstitutiveModelFn, StaticMap},
};
use nalgebra::{RowSVector, SMatrix, SVector};

pub struct PlasticityState<
    const STRESS_STRAIN: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    const KAPPA: usize,
> {
    pub f: f64,
    pub df_dsigma: RowSVector<f64, STRESS_STRAIN>,
    pub df_dkappa: RowSVector<f64, KAPPA>,
    pub g: SVector<f64, STRESS_STRAIN>,
    pub dg_dkappa: SMatrix<f64, STRESS_STRAIN, KAPPA>,
    pub dg_dsigma: SMatrix<f64, STRESS_STRAIN, STRESS_STRAIN>,
    pub k: SVector<f64, KAPPA>,
    pub dk_dsigma: SMatrix<f64, KAPPA, STRESS_STRAIN>,
    pub dk_dkappa: SMatrix<f64, KAPPA, KAPPA>,
    //pub elastic_tangent: SMatrix<f64, STRESS_STRAIN, STRESS_STRAIN>,
    //pub elastic_tangent_inv: SMatrix<f64, STRESS_STRAIN, STRESS_STRAIN>,
}
impl<
    const STRESS_STRAIN: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    const KAPPA: usize,
> PlasticityState<STRESS_STRAIN, N_PARAMETERS, PARAMETERS, KAPPA>
{
    pub fn zeros() -> PlasticityState<STRESS_STRAIN, N_PARAMETERS, PARAMETERS, KAPPA> {
        PlasticityState {
            f: 0.0,
            df_dsigma: RowSVector::zeros(),
            df_dkappa: RowSVector::zeros(),
            g: SVector::zeros(),
            dg_dkappa: SMatrix::zeros(),
            dg_dsigma: SMatrix::zeros(),
            k: SVector::zeros(),
            dk_dsigma: SMatrix::zeros(),
            dk_dkappa: SMatrix::zeros(),
            //elastic_tangent: SMatrix::zeros(),
            //elastic_tangent_inv: SMatrix::zeros(),
        }
    }
}

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
        &self,
        sigma: &SVector<f64, STRESS_STRAIN>,
        kappa: &SVector<f64, KAPPA>,
        state: &mut PlasticityState<STRESS_STRAIN, N_PARAMETERS, PARAMETERS, KAPPA>,
    );
    fn calculate_f_only(&self, sigma: &SVector<f64, STRESS_STRAIN>, kappa: &SVector<f64, KAPPA>) -> f64;
    // fn f(&self) -> f64;
    // fn df_dsigma(&self) -> &RowSVector<f64, STRESS_STRAIN>;
    // fn df_dkappa(&self) -> &RowSVector<f64, KAPPA>;
    // fn g(&self) -> &SVector<f64, STRESS_STRAIN>;
    // fn dg_dkappa(&self) -> &SMatrix<f64, STRESS_STRAIN, KAPPA>;
    // fn dg_dsigma(&self) -> &SMatrix<f64, STRESS_STRAIN, STRESS_STRAIN>;
    // fn k(&self) -> &SVector<f64, KAPPA>;
    // fn dk_dsigma(&self) -> &SMatrix<f64, KAPPA, STRESS_STRAIN>;
    // fn dk_dkappa(&self) -> &SMatrix<f64, KAPPA, KAPPA>;
    fn elastic_tangent(&self) -> &SMatrix<f64, STRESS_STRAIN, STRESS_STRAIN>;
    fn elastic_tangent_inv(&self) -> &SMatrix<f64, STRESS_STRAIN, STRESS_STRAIN>;
    /// Stress used to nondimensionalize the yield function in the return mapping, e.g. the
    /// initial yield stress. Defaults to the magnitude of the trial stress, which is nonzero
    /// whenever the trial state is plastic.
    fn characteristic_stress(&self, sigma_tr: &SVector<f64, STRESS_STRAIN>) -> f64 {
        sigma_tr.norm().max(f64::MIN_POSITIVE)
    }
    /// Whether the return mapping may evaluate the model at `kappa`. Trial points of the
    /// line search that are not admissible are rejected before evaluation.
    fn is_admissible(&self, _kappa: &SVector<f64, KAPPA>) -> bool {
        true
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
        let model = MODEL::new(parameters_);
        //let mut state = PlasticityState::zeros();
        let elastic_tangent = model.elastic_tangent();

        let del_eps = SVector::<f64, 6>::from_column_slice(del_strain);
        let sigma_0 = SVector::<f64, 6>::from_column_slice(stress);
        let sigma_tr = elastic_tangent * del_eps + &sigma_0;

        let alpha_0 = SVector::<f64, 1>::from_element(history_.alpha);
        //let mut alpha_1 = alpha_0.clone();
        //let mut sigma_1: SVector<f64, 6>;
        //let state = model.set_model_state(&sigma_tr, &alpha_0);

        let f = model.calculate_f_only(&sigma_tr, &alpha_0);
        if f <= 0.0 {
            *stress = sigma_tr.data.0[0];
            if let Some(tangent) = tangent {
                *tangent = model.elastic_tangent().data.0;
            }
            return;
        } else {
            let solver = NewtonSolverStrain::<8, 6, N_PARAMETERS, PARAMETERS, 1, MODEL>::new(
                &model, 1e-12, 1e-12, 50,
            );
            let mut state = PlasticityState::zeros();

            let result = solver
                .solve(&sigma_tr, &alpha_0, &mut state)
                .unwrap_or_else(|err| {
                    panic!(
                        "Plasticity failed to converge on sigma_tr {}, sigma_0 {}, alpha_0 {}: {:?}",
                        sigma_tr, sigma_0, alpha_0, err
                    )
                });

            // Update the stress and history
            *stress = result.sigma.data.0[0];
            history_.alpha = result.kappa[0];
            history_.plastic_strain += result.del_plastic_strain;
            if let Some(tangent) = tangent {
                //update the newton matrix in order to evaluate at t_{n+1}
                let mut dres = SMatrix::<f64, 8, 8>::zeros();
                stress_based_jacobian(
                    model.elastic_tangent(),
                    &state,
                    &mut dres,
                    result.del_lambda,
                );
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
