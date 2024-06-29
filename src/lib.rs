#![feature(assert_matches)]
#![feature(adt_const_params)]
#![feature(generic_arg_infer)]
#![feature(generic_const_exprs)]
#![feature(trait_upcasting)]
#![feature(const_trait_impl)]
#![feature(concat_idents)]
#![feature(associated_const_equality)]
#![allow(incomplete_features)]

#[macro_use]
extern crate tensrus_derive;

mod blas;
pub mod differentiable;
pub mod dyn_tensor;
pub mod encoding;
pub mod errors;
pub mod generic_tensor;
pub mod iterator;
pub mod matrix;
pub mod op;
pub mod scalar;
pub mod shape;
pub mod storage;
pub mod tensor;
pub mod translation;
pub mod type_assert;
pub mod var;
pub mod vector;
pub mod view;
