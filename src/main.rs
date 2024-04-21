pub mod matrix;
pub mod nn;
pub mod numeric;
pub mod render;
pub mod scalar;
pub mod tensor;
pub mod value;
pub mod vector;

use nn::{Module, MLP};
use render::Graph;
use std::fs::File;
use std::process::Command;
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

    let n: MLP<f64> = MLP::new(3, vec![1]);

    for i in 0..1 {
        let ypred: Vec<Value<f64>> = xs.iter().map(|x| n.call(x.to_vec())[0].clone()).collect();
        let loss = n.loss(&ys, &ypred);
        render_graph(loss.clone(), 1);

        n.zero_grad();
        loss.backward();
        render_graph(loss.clone(), 2);
        for p in n.parameters() {
            p.update_from_grad(0.01);
        }
        render_graph(loss.clone(), 3);

        println!("{}: Loss: {:#?}", i, loss);
    }
}

fn render_graph(x: Value<f64>, id: usize) {
    let g = Graph::new(x.clone());
    let dotfile = format!("/tmp/{}.dot", id);
    let mut f = File::create(&dotfile).unwrap();
    g.render_to(&mut f);
    let pngfile = format!("/tmp/{}.png", id);
    Command::new("dot")
        .args(["-Tpng", format!("-o{}", pngfile).as_ref(), &dotfile])
        .output()
        .unwrap();
    println!("rendered {}", pngfile);
}
