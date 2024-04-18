#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

pub mod tensor;
pub mod matrix;
pub mod vector;
pub mod scalar;

pub use tensor::Tensor;
