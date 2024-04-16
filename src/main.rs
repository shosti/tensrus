#![feature(generic_const_exprs)]

mod tensor;

fn main() {
    let x = tensor::Value::from(1.23);
    println!("X: {:#?}", x);

    let y = tensor::Vector::from([1.23, 4.56]);
    println!("Y: {:#?}", y);

    let z = tensor::Matrix::from([[1.23, 4.56, 7.89], [2.34, 5.67, 8.90]]);
    println!("Z: {:#?}", z);
}
