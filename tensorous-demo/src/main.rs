use tensorous::nn::{Module, MLP};
use tensorous::scalar::Scalar;
use tensorous::var::Var;

fn main() {
    let xs: Vec<Vec<Var<Scalar<f64>>>> = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    .iter()
    .map(|row| row.iter().map(|&n| Var::from(n)).collect())
    .collect();

    let ys: Vec<Var<Scalar<f64>>> = [1.0, -1.0, -1.0, 1.0]
        .iter()
        .map(|&n| Var::from(n))
        .collect();

    let n: MLP<f64> = MLP::new(3, vec![4, 4, 1]);

    let mut ypred: Vec<Var<Scalar<f64>>>;
    for i in 0..50 {
        ypred = xs.iter().map(|x| call(x.to_vec())[0].clone()).collect();
        let loss = n.loss(&ys, &ypred);
        loss.backward().unwrap();
        for p in n.parameters() {
            p.update_from_grad(0.1).unwrap();
        }
        println!("{} LOSS: {:#?}", i, loss.data().val());
    }
}
