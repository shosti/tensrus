pub mod matrix;
pub mod nn;
pub mod numeric;
pub mod scalar;
pub mod tensor;
pub mod value;
pub mod vector;

use nn::{Module, MLP};
pub use tensor::Tensor;
use value::Value;

fn main() {
    let xs: Vec<Vec<Value<f64>>> = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    .iter()
    .map(|row| row.iter().map(|&n| Value::from(n)).collect())
    .collect();

    let ys: Vec<Value<f64>> = [1.0, -1.0, 1.0, 1.0]
        .iter()
        .map(|&n| Value::from(n))
        .collect();

    println!("XS: {:#?}", xs);
    println!("YS: {:#?}", ys);
    let n = MLP::new(3, vec![4, 4, 1]);
    for _ in 0..1 {
        let ypred: Vec<Value<f64>> = xs.iter().map(|x| n.call(x.to_vec())[0].clone()).collect();
        println!("YPRED: {:#?}", ypred);

        let loss: Value<f64> = std::iter::zip(&ys, &ypred)
            .map(|(ygt, yout)| (yout.clone() - ygt.clone()).pow(2.0))
            .sum();
        println!("LOSS: {:#?}", loss);
        loss.backward();
        for p in n.parameters().iter() {
            p.update_from_grad(0.01);
        }
    }
}
