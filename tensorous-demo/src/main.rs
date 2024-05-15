use tensorous::flow::Flow;
use tensorous::nn::{Module, MLP};
use tensorous::scalar::Scalar;

fn main() {
    let xs: Vec<Vec<Flow<Scalar<f64>>>> = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    .iter()
    .map(|row| row.iter().map(|&n| Flow::new(Scalar::from(n))).collect())
    .collect();

    let ys: Vec<Flow<Scalar<f64>>> = [1.0, -1.0, -1.0, 1.0]
        .iter()
        .map(|&n| Flow::from(n))
        .collect();

    let mut n: MLP<f64> = MLP::new(3, vec![4, 4, 1]);

    let mut ypred;
    for i in 0..50 {
        ypred = xs.iter().map(|x| n.call(x.to_vec())[0].clone()).collect();
        let mut loss = n.loss(&ys, &ypred);
        n.zero_grad();
        loss.backward();
        for p in n.parameters() {
            p.update_from_grad(0.1);
        }
        println!("{} LOSS: {:#?}", i, loss.val());
    }
}