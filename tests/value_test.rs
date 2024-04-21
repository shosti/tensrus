extern crate brainy;

use brainy::value::Value;
use brainy::render::Graph;
use std::fs::File;
use std::process::Command;

#[test]
fn value_sanity_test() {
    let x = Value::from(-4.0);
    let z = Value::from(2.0) * x.clone() + Value::from(2.0) + x.clone();
    let q = z.clone().relu() + z.clone() * x.clone();
    let h = (z.clone() * z.clone()).relu();
    let y = h.clone() + q.clone() + q.clone() * x.clone();
    y.backward();
    render_graph(y.clone(), "thing".to_string());

    assert_eq!(y.val(), -20.0);
    assert_eq!(x.grad(), 46.0);
}

fn render_graph(x: Value<f64>, id: String) {
    let g = Graph::new(x.clone());
    let dotfile = format!("/tmp/{}.dot", id);
    let mut f = File::create(&dotfile).unwrap();
    g.render_to(&mut f);
    let pdffile = format!("/tmp/{}.pdf", id);
    Command::new("dot")
        .args(["-Tpdf", format!("-o{}", pdffile).as_ref(), &dotfile])
        .output()
        .unwrap();
    println!("rendered {}", pdffile);
}
