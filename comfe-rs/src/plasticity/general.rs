use std::marker::PhantomData;

use crate::QDim;
use crate::linear_elasticity::IsotropicElasticParameters;
use crate::plasticity::{NewtonSolverStrain, stress_based_jacobian};
use crate::{
    create_history_parameter_struct,
    interfaces::{ArrayEquivalent, ConstitutiveModelFn, StaticMap},
};
use nalgebra::{RowSVector, SMatrix, SVector};

pub struct PlasticityState<const STRESS_STRAIN: usize, const KAPPA: usize> {
    pub f: f64,
    pub df_dsigma: RowSVector<f64, STRESS_STRAIN>,
    pub df_dkappa: RowSVector<f64, KAPPA>,
    pub g: SVector<f64, STRESS_STRAIN>,
    pub dg_dkappa: SMatrix<f64, STRESS_STRAIN, KAPPA>,
    pub dg_dsigma: SMatrix<f64, STRESS_STRAIN, STRESS_STRAIN>,
    pub k: SVector<f64, KAPPA>,
    pub dk_dsigma: SMatrix<f64, KAPPA, STRESS_STRAIN>,
    pub dk_dkappa: SMatrix<f64, KAPPA, KAPPA>,
}
impl<const STRESS_STRAIN: usize, const KAPPA: usize> PlasticityState<STRESS_STRAIN, KAPPA> {
    pub fn zeros() -> PlasticityState<STRESS_STRAIN, KAPPA> {
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
        state: &mut PlasticityState<STRESS_STRAIN, KAPPA>,
    );
    fn calculate_f_only(&self, sigma: &SVector<f64, STRESS_STRAIN>, kappa: &SVector<f64, KAPPA>) -> f64;
    /// Stress used to nondimensionalize the yield function in the return mapping, e.g. the
    /// initial yield stress. Defaults to the magnitude of the trial stress, which is nonzero
    /// whenever the trial state is plastic.
    fn characteristic_stress(&self, sigma_tr: &SVector<f64, STRESS_STRAIN>) -> f64 {
        sigma_tr.norm().max(f64::MIN_POSITIVE)
    }
    /// Whether the return mapping may evaluate the model at `sigma` and `kappa`, e.g. not past
    /// a non-differentiable point of the yield surface. Trial points of the line search that are
    /// not admissible are rejected before evaluation and the step is halved.
    fn is_admissible(
        &self,
        _sigma: &SVector<f64, STRESS_STRAIN>,
        _kappa: &SVector<f64, KAPPA>,
    ) -> bool {
        true
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
where
    MODEL::Parameters: IsotropicElasticParameters,
{
    type History = IsotropicPlasticityHistory3D;
    type Parameters = MODEL::Parameters;
    #[inline]
    fn evaluate(
        _time: f64,
        _del_time: f64,
        del_strain: &[f64; 6],
        stress: &mut [f64; 6],
        tangent: Option<&mut [[f64; 6]; 6]>,
        history: &mut [f64; 7],
        parameters: &[f64; PARAMETERS],
    ) {
        let parameters_ = Self::Parameters::from_array(parameters);
        let history_ = IsotropicPlasticityHistory3D::from_array_mut(history);
        let elastic_tangent = parameters_.elastic_tangent::<6>();
        let model = MODEL::new(parameters_);

        let del_eps = SVector::<f64, 6>::from_column_slice(del_strain);
        let sigma_0 = SVector::<f64, 6>::from_column_slice(stress);
        let sigma_tr = &elastic_tangent * del_eps + &sigma_0;

        let alpha_0 = SVector::<f64, 1>::from_element(history_.alpha);

        let f = model.calculate_f_only(&sigma_tr, &alpha_0);
        if f <= 0.0 {
            *stress = sigma_tr.data.0[0];
            if let Some(tangent) = tangent {
                *tangent = elastic_tangent.data.0;
            }
            return;
        } else {
            let solver = NewtonSolverStrain::<8, 6, N_PARAMETERS, PARAMETERS, 1, MODEL>::new(
                &model, &elastic_tangent, 1e-12, 1e-12, 50,
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
                let mut plastic_tangent =
                    isotropic_algorithmic_tangent(&elastic_tangent, &state, result.del_lambda)
                        .expect("Plasticity3D: Failed to calculate tangent");
                plastic_tangent.transpose_mut(); //TODO: move the transpose to the python bindings
                *tangent = plastic_tangent.data.0;
            }
        }
    }
}

/// Algorithmic tangent $\partial\sigma/\partial\sigma_{tr} \cdot C = \partial\sigma/\partial\varepsilon_{tr}$
/// of a converged return mapping (not transposed, entry $(i,j)$ is
/// $\partial\sigma_i/\partial\varepsilon_j$). `state` must be the plasticity state at the converged
/// stress and kappa, as left by [`NewtonSolverStrain::solve`], and `elastic_tangent` the elastic
/// tangent used in the return mapping. Returns `None` if the Newton matrix is singular.
pub fn isotropic_algorithmic_tangent(
    elastic_tangent: &SMatrix<f64, 6, 6>,
    state: &PlasticityState<6, 1>,
    del_lambda: f64,
) -> Option<SMatrix<f64, 6, 6>> {
    let mut dres = SMatrix::<f64, 8, 8>::zeros();
    stress_based_jacobian(elastic_tangent, state, &mut dres, del_lambda);
    let inverse = dres.try_inverse()?;
    Some(inverse.fixed_view::<6, 6>(0, 0) * elastic_tangent)
}
