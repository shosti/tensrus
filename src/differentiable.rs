use std::{
    iter::Sum,
    ops::{Add, Mul},
};

use crate::{
    blas::BLASOps,
    dyn_tensor::{DynTensor, FromDynTensor},
    tensor::Tensor,
};

pub trait Differentiable: num::Float + Copy + BLASOps + std::fmt::Debug + Sum + 'static {
    fn two() -> Self {
        Self::one() + Self::one()
    }
}

pub trait DifferentiableTensor:
    Tensor
    + DynTensor
    + FromDynTensor
    + Clone
    + for<'a> Add<&'a Self, Output = Self>
    + Mul<Self::T, Output = Self>
    + 'static
where
    Self::T: Differentiable,
{
}

impl Differentiable for f32 {}
impl Differentiable for f64 {}
