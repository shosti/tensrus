#![feature(generic_const_exprs)]

mod tensor;

use crate::tensor::Tensor;

fn main() {
    let x = tensor::Value::from(1.23);
    println!("X: {:#?}", x);
    println!("X.DIM(): {:#?}", x.dim());
    println!("X.SHAPE(): {:#?}", x.shape());

    let y = tensor::Vector::from([1.23, 4.56]);
    println!("Y: {:#?}", y);
    println!("Y.DIM(): {:#?}", y.dim());
    println!("Y.SHAPE(): {:#?}", y.shape());

    let z = tensor::Matrix::from([[1.23, 4.56, 7.89], [2.34, 5.67, 8.90]]);
    println!("Z: {:#?}", z);
    println!("Z.DIM(): {:#?}", z.dim());
    println!("Z.SHAPE(): {:#?}", z.shape());
}
