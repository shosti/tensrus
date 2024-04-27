#![feature(adt_const_params)]
#![feature(generic_arg_infer)]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#[macro_use]
extern crate brainy_derive;

pub mod flow;
pub mod generic_tensor;
pub mod matrix;
pub mod nn;
pub mod numeric;
pub mod op;
pub mod render;
pub mod scalar;
pub mod tensor;
pub mod value;
pub mod vector;

// use nn::{Module, MLP};
// use render::Graph;
// use std::fs::File;
// use std::process::Command;
// pub use tensor::GenericTensor;
// use value::Value;

// fn main() {
//     let xs: Vec<Vec<Value<f64>>> = [
//         [2.0, 3.0, -1.0],
//         [3.0, -1.0, 0.5],
//         [0.5, 1.0, 1.0],
//         [1.0, 1.0, -1.0],
//     ]
//     .iter()
//     .map(|row| row.iter().map(|&n| Value::from(n)).collect())
//     .collect();

//     let ys: Vec<Value<f64>> = [1.0, -1.0, -1.0, 1.0]
//         .iter()
//         .map(|&n| Value::from(n))
//         .collect();

//     let n: MLP<f64> = MLP::new(3, vec![4, 4, 1]);

//     let mut ypred;
//     for i in 0..50 {
//         ypred = xs.iter().map(|x| n.call(x.to_vec())[0].clone()).collect();
//         let loss = n.loss(&ys, &ypred);
//         n.zero_grad();
//         loss.backward();
//         for p in n.parameters() {
//             p.update_from_grad(0.1);
//         }
//         println!("{} LOSS: {:#?}", i, loss);
//     }
// }
