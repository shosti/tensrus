#![feature(generic_const_exprs)]

mod tensor;

use crate::tensor::Tensor;

fn main() {
    let x = tensor::Scalar::from(1.23);
    println!("X: {:#?}", x);
    println!("X.RANK(): {:#?}", x.rank());
    println!("X.SHAPE(): {:#?}", x.shape());

    let y = tensor::Vector::from([1.23, 4.56]);
    println!("Y: {:#?}", y);
    println!("Y.RANK(): {:#?}", y.rank());
    println!("Y.SHAPE(): {:#?}", y.shape());

    let mut z = tensor::Matrix::from([[1.23, 4.56, 7.89], [2.34, 5.67, 8.90]]);
    println!("Z: {:#?}", z);
    println!("Z.RANK(): {:#?}", z.rank());
    println!("Z.SHAPE(): {:#?}", z.shape());

    z *= 7.0;
    println!("Z: {:#?}", z);
}
