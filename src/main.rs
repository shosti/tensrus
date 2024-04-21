pub mod matrix;
pub mod nn;
pub mod numeric;
pub mod scalar;
pub mod tensor;
pub mod value;
pub mod vector;
pub mod render;

// use nn::{Module, MLP};
pub use tensor::Tensor;
use value::Value;
 use render::Graph;

fn main() {
    let x1 = Value::from(2.0);
    let x2 = Value::from(0.0);
    let w1 = Value::from(-3.0);
    let w2 = Value::from(1.0);
    let b = Value::from(6.8);

    let x1w1 = x1 * w1;
    let x2w2 = x2 * w2;
    let x1w1x2w2 = x1w1 + x2w2;
    let n = x1w1x2w2 + b;
    let o = n.relu();

    let (nodes, edges) = o.trace();
    let g: Graph = Graph::new(nodes, edges);
    let mut stdout = std::io::stdout().lock();
    g.render_to(&mut stdout)

    // let xs: Vec<Vec<Value<f64>>> = [
    //     [2.0, 3.0, -1.0],
    //     [3.0, -1.0, 0.5],
    //     [0.5, 1.0, 1.0],
    //     [1.0, 1.0, -1.0],
    // ]
    // .iter()
    // .map(|row| row.iter().map(|&n| Value::from(n)).collect())
    // .collect();

    // let ys: Vec<Value<f64>> = [1.0, -1.0, 1.0, 1.0]
    //     .iter()
    //     .map(|&n| Value::from(n))
    //     .collect();

    // let n = MLP::new(3, vec![4, 4, 1]);
    // let ypred: Vec<Value<f64>> = xs.iter().map(|x| n.call(x.to_vec())[0].clone()).collect();
    // let loss: Value<f64> = std::iter::zip(&ys, &ypred)
    //     .map(|(ygt, yout)| (yout.clone() - ygt.clone()).pow(2.0))
    //     .sum();
    // let (nodes, edges) = loss.trace();
    // draw_dot(nodes, edges);
    // println!("XS: {:#?}", xs);
    // println!("YS: {:#?}", ys);
    // for _ in 0..1 {
    //     let ypred: Vec<Value<f64>> = xs.iter().map(|x| n.call(x.to_vec())[0].clone()).collect();
    //     println!("YPRED: {:#?}", ypred);

    //     println!("LOSS: {:#?}", loss);
    //     loss.backward();
    //     for p in n.parameters().iter() {
    //         p.update_from_grad(0.01);
    //     }
    // }
}
