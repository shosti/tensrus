#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod tensor;
mod matrix;
mod vector;
mod scalar;

use crate::tensor::Tensor;

fn main() {
    let x = scalar::Scalar::from(1.23);
    println!("X: {:#?}", x);
    println!("X.RANK(): {:#?}", x.rank());
    println!("X.SHAPE(): {:#?}", x.shape());

    let y = vector::Vector::from([1.23, 4.56]);
    println!("Y: {:#?}", y);
    println!("Y.RANK(): {:#?}", y.rank());
    println!("Y.SHAPE(): {:#?}", y.shape());

    let mut z = matrix::Matrix::from([[1.23, 4.56, 7.89], [2.34, 5.67, 8.90]]);
    println!("Z: {:#?}", z);
    println!("Z.RANK(): {:#?}", z.rank());
    println!("Z.SHAPE(): {:#?}", z.shape());

    z *= 7.0;
    println!("Z: {:#?}", z);
}
