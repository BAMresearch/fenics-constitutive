use std::collections::HashMap;
use std::num::NonZeroUsize;

use crate::{
    consts::*,
    mandel::{MandelView, MandelViewMut, nonsymmetric_tensor_to_mandel},
};
use konst::{const_eq, eq_str};
use nalgebra::{SMatrix, SMatrixViewMut, SVector, SVectorView, SVectorViewMut, Scalar};

pub enum StressStrainConstraint {
    UNIAXIAL_STRAIN = 1,
    UNIAXIAL_STRESS = 2,
    PLANE_STRAIN = 3,
    PLANE_STRESS = 4,
    FULL = 5,
}
impl StressStrainConstraint {
    pub const fn stress_strain_dim(&self) -> usize {
        match self {
            StressStrainConstraint::UNIAXIAL_STRAIN => 1,
            StressStrainConstraint::UNIAXIAL_STRESS => 1,
            StressStrainConstraint::PLANE_STRAIN => 4,
            StressStrainConstraint::PLANE_STRESS => 4,
            StressStrainConstraint::FULL => 6,
        }
    }
    pub const fn geometric_dim(&self) -> usize {
        match self {
            StressStrainConstraint::UNIAXIAL_STRAIN => 1,
            StressStrainConstraint::UNIAXIAL_STRESS => 1,
            StressStrainConstraint::PLANE_STRAIN => 2,
            StressStrainConstraint::PLANE_STRESS => 2,
            StressStrainConstraint::FULL => 3,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub enum QDim {
    Scalar,
    Vector(usize),
    Matrix(usize),
    RotatableVector(usize),
    RotatableMatrix(usize),
}
impl QDim {
    pub const fn dim(&self) -> usize {
        match self {
            QDim::Scalar => 1,
            QDim::Vector(size) => *size,
            QDim::RotatableVector(size) => *size,
            QDim::Matrix(size) => *size,
            QDim::RotatableMatrix(size) => *size,
        }
    }
    pub const fn size(&self) -> usize {
        match self {
            QDim::Scalar => 1,
            QDim::Vector(size) => *size,
            QDim::RotatableVector(size) => *size,
            QDim::Matrix(size) => *size * *size,
            QDim::RotatableMatrix(size) => *size * *size,
        }
    }
}

#[macro_export]
macro_rules! q_dim_data_type {
    ((QDim::Scalar)) => {
        f64
    };
    ((QDim::Vector($size:expr))) => {
        SVector<f64, $size>
    };
    ((QDim::RotatableVector($size:expr))) => {
        SVector<f64, $size>
    };
    ((QDim::Matrix($size:expr))) => {
        SMatrix<f64, $size, $size>
    };
    ((QDim::RotatableMatrix($size:expr))) => {
        SMatrix<f64, $size, $size>
    };
}
pub trait ArrayEquivalent<const N: usize>: Sized {
    fn from_array(array: &[f64; N]) -> &Self;
    fn from_array_mut(array: &mut [f64; N]) -> &mut Self;

    fn as_array(&self) -> &[f64; N];
    fn as_array_mut(&mut self) -> &mut [f64; N];
}
pub trait StaticMap<const N: usize, T>
where
    T: Copy,
{
    const MAP: [(&'static str, T); N];

    //Todo: When constant traits become a thing, make this a constant function
    fn get(name: &str) -> Option<T> {
        for i in 0..N {
            if eq_str(Self::MAP[i].0, name) {
                return Some(Self::MAP[i].1);
            }
        }
        None
    }
    fn get_index(name: &str) -> Option<usize> {
        for i in 0..N {
            if eq_str(Self::MAP[i].0, name) {
                return Some(i);
            }
        }
        None
    }
}

impl<const N: usize> ArrayEquivalent<N> for [f64; N] {
    #[inline]
    fn from_array(array: &[f64; N]) -> &Self {
        array
    }

    #[inline]
    fn from_array_mut(array: &mut [f64; N]) -> &mut Self {
        array
    }

    #[inline]
    fn as_array(&self) -> &[f64; N] {
        self
    }

    #[inline]
    fn as_array_mut(&mut self) -> &mut [f64; N] {
        self
    }
}
impl<const N: usize> StaticMap<1, QDim> for [f64; N] {
    const MAP: [(&'static str, QDim); 1] = [("all_fields", QDim::Vector(N))];
}

#[macro_export]
macro_rules! create_history_parameter_struct {
    ($(#[$doc:meta])* $struct_name:ident, $n:expr, $size:expr, [$(($field_name:ident, $qdim:tt)),*]) => {
        $(#[$doc])*
        #[repr(C)]
        #[derive(Copy, Clone, Debug, Default)]
        pub struct $struct_name {
            $(
                pub $field_name: q_dim_data_type!($qdim),
            )*
        }

        impl StaticMap<$n, QDim> for $struct_name {
            const MAP: [(&'static str, QDim); $n] = [
                $((stringify!($field_name), $qdim)),*
            ];
        }
        impl_array_equivalent!($struct_name, $size);
    };
}

#[macro_export]
macro_rules! impl_array_equivalent {
    ($type:ty, $size:expr) => {
        const _: () = assert!(
            std::mem::size_of::<[f64; $size]>() == std::mem::size_of::<$type>(),
            "size mismatch"
        );

        impl ArrayEquivalent<$size> for $type {
            #[inline]
            fn from_array(array: &[f64; $size]) -> &Self {
                unsafe { &*(array as *const [f64; $size] as *const Self) }
            }

            #[inline]
            fn from_array_mut(array: &mut [f64; $size]) -> &mut Self {
                unsafe { &mut *(array as *mut [f64; $size] as *mut Self) }
            }

            #[inline]
            fn as_array(&self) -> &[f64; $size] {
                unsafe { &*(self as *const Self as *const [f64; $size]) }
            }

            #[inline]
            fn as_array_mut(&mut self) -> &mut [f64; $size] {
                unsafe { &mut *(self as *mut Self as *mut [f64; $size]) }
            }
        }
    };
}
pub trait ConstitutiveModelFn<
    const STRESS_STRAIN: usize,
    const N_HISTORY: usize,
    const HISTORY: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
> where
    Self: Sized,
{
    type History: ArrayEquivalent<HISTORY> + StaticMap<N_HISTORY, QDim>;
    type Parameters: ArrayEquivalent<PARAMETERS> + StaticMap<N_PARAMETERS, QDim>;

    const STRESS_STRAIN: usize = STRESS_STRAIN;
    const N_HISTORY: usize = N_HISTORY;
    const HISTORY: usize = HISTORY;
    const N_PARAMETERS: usize = N_PARAMETERS;
    const PARAMETERS: usize = PARAMETERS;

    fn evaluate(
        time: f64,
        del_time: f64,
        //strain: &[f64; STRESS_STRAIN],
        del_strain: &[f64; STRESS_STRAIN],
        stress: &mut [f64; STRESS_STRAIN],
        tangent: Option<&mut [[f64; STRESS_STRAIN]; STRESS_STRAIN]>,
        history: &mut [f64; HISTORY],
        parameters: &[f64; PARAMETERS],
    );

    //fn evaluate_all(
    //    time: f64,
    //    del_time: f64,
    //    //strain: &[f64],
    //    del_strain: &[f64],
    //    stress: &mut [f64],
    //    tangent: Option<&mut [f64]>,
    //    history: &mut [f64],
    //    parameters: &[f64],
    //) {
    //    evaluate_model::<STRESS_STRAIN, N_HISTORY, HISTORY, N_PARAMETERS, PARAMETERS, Self>(
    //        time, del_time, del_strain, stress, tangent, history, parameters
    //    );
    //}
}

trait SmallStrainConstitutiveModel<
    const STRESS_STRAIN: usize,
    const N_HISTORY: usize,
    const HISTORY: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
> where
    Self: Sized,
{
    type History: ArrayEquivalent<HISTORY> + StaticMap<N_HISTORY, QDim>;
    type Parameters: ArrayEquivalent<PARAMETERS> + StaticMap<N_PARAMETERS, QDim>;

    const STRESS_STRAIN: usize = STRESS_STRAIN;
    const N_HISTORY: usize = N_HISTORY;
    const HISTORY: usize = HISTORY;
    const N_PARAMETERS: usize = N_PARAMETERS;
    const PARAMETERS: usize = PARAMETERS;

    fn evaluate(
        time: f64,
        del_time: f64,
        //strain: &SVector<f64, STRESS_STRAIN>,
        del_strain: &SVector<f64, STRESS_STRAIN>,
        stress: &mut SVector<f64, STRESS_STRAIN>,
        tangent: Option<&mut SMatrix<f64, STRESS_STRAIN, STRESS_STRAIN>>,
        history: &mut Self::History,
        parameters: &Self::Parameters,
    );

    fn evaluate_all(
        time: f64,
        del_time: f64,
        //strain: &[f64],
        del_strain: &[f64],
        stress: &mut [f64],
        tangent: Option<&mut [f64]>,
        history: &mut [f64],
        parameters: &[f64],
    );
}

/// A function that checks if the combined length of the history and the parameters is equal to `HISTORY` and `PARAMETERS`.
/// This is needed because both the number of history values and parameters and the size of both (which are different if we have
/// for example one vector-valued history variable) are defined seperately.
pub const fn check_constitutive_model_maps<
    const STRESS_STRAIN: usize,
    const N_HISTORY: usize,
    const HISTORY: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    T: ConstitutiveModelFn<STRESS_STRAIN, N_HISTORY, HISTORY, N_PARAMETERS, PARAMETERS>,
>() -> bool {
    let parameters = T::Parameters::MAP;
    let history = T::History::MAP;
    let mut i: usize = 0;
    let mut size_parameters: usize = 0;
    let mut size_history: usize = 0;
    while i < N_HISTORY {
        size_history += history[i].1.size();
        i += 1;
    }
    i = 0;
    while i < N_PARAMETERS {
        size_parameters += parameters[i].1.size();
        i += 1;
    }
    const_eq!(size_parameters, PARAMETERS) && const_eq!(size_history, HISTORY)
}

/// Evaluates a constitutive model with input for more than one quadrature point.
/// Panics if the sizes of the input are inconsistent.
pub fn evaluate_model<
    const STRESS_STRAIN: usize,
    const GEOMETRY: usize,
    const N_HISTORY: usize,
    const HISTORY: usize,
    const N_PARAMETERS: usize,
    const PARAMETERS: usize,
    MODEL: ConstitutiveModelFn<STRESS_STRAIN, N_HISTORY, HISTORY, N_PARAMETERS, PARAMETERS> + Sized,
>(
    time: f64,
    del_time: f64,
    //strain: &[f64],
    del_grad_u: &[f64],
    stress: &mut [f64],
    tangent: Option<&mut [f64]>,
    history: &mut [f64],
    parameters: &[f64],
) {
    let parameters: [f64; PARAMETERS] = parameters.try_into().expect(&format!(
        "Length of parameters slice does not match the expected length. Expected: {}, got: {}.",
        PARAMETERS,
        parameters.len()
    ));

    let (stress_, stress_rest) = stress.as_chunks_mut::<STRESS_STRAIN>();
    //let (strain_, strain_rest) = strain.as_chunks::<STRESS_STRAIN>();
    let (del_grad_u_, del_grad_u_rest) = del_grad_u.as_chunks::<GEOMETRY>();
    let (del_grad_u_, del_grad_u_rest) = del_grad_u_.as_chunks::<GEOMETRY>();

    let mut tangent_ = {
        match tangent {
            Some(t) => {
                let (tangent_, tangent_rest_1) = t.as_chunks_mut::<STRESS_STRAIN>();
                
                let (tangent_, tangent_rest_2) = tangent_.as_chunks_mut::<STRESS_STRAIN>();
                assert!(tangent_rest_1.is_empty() && tangent_rest_2.is_empty());
                Some(tangent_)
            }
            None => None,
        }
    };

    let stress_len = stress_.len();
    //let del_strain_len = del_strain_.len();
    let del_grad_u_len = del_grad_u_.len();
    //let history_len = history_.len();
    let tangent_len = tangent_.as_ref().map_or(0, |t| t.len());

    assert!(
        stress_len == del_grad_u_len
            //&& stress_len == strain_len
            //&& stress_len == history_len
            && (stress_len == tangent_len || tangent_.is_none()),
        "Stress, strain, and tangent lengths do not match: \
        stress_len: {}, del_grad_u_len: {}, tangent_len: {}",
        stress_len,
        //strain_len,
        del_grad_u_len,
        //history_len,
        tangent_len
    );

    //deal with special case that history is zero-sized
    let mut zero_history: Vec<[f64; HISTORY]> = vec![];
    if HISTORY == 0 {
        zero_history.resize(stress_len, [0.0; HISTORY]);
    }
    let (history_, history_rest) = {
        if HISTORY == 0 {
            //let history_chunks : &mut [[f64; HISTORY];0] =  &mut[];
            let history_rest: &mut [f64] = &mut [];
            (zero_history.as_mut_slice(), history_rest)
        } else {
            history.as_chunks_mut::<HISTORY>()
        }
    };
    assert!(stress_len== history_.len());


    assert!(
        stress_rest.is_empty()
            && del_grad_u_rest.is_empty()
            && history_rest.is_empty(),
        "Input slices are not of the correct length: \
        stress_rest: {:?}, del_grad_u_rest: {:?}, history_rest: {:?}",
        stress_rest.len(),
        del_grad_u_rest.len(),
        history_rest.len()
    );

    for i in 0..stress_len {
        let tangent_chunk: Option<&mut [[f64; STRESS_STRAIN]; STRESS_STRAIN]> =
            tangent_.as_mut().map(|t| &mut t[i]);
        let del_strain_chunk: [f64; STRESS_STRAIN] = nonsymmetric_tensor_to_mandel(del_grad_u_[i]);
        MODEL::evaluate(
            time,
            del_time,
            //&strain_[i],
            &del_strain_chunk,
            &mut stress_[i],
            tangent_chunk,
            &mut history_[i],
            &parameters,
        );
    }
}
