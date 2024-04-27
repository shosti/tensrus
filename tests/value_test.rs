extern crate brainy;

use brainy::render::Graph;
use brainy::value::Value;
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

#[test]
fn test_more_ops() {
    let a = Value::from(-4.0);
    let b = Value::from(2.0);
    let mut c = a.clone() + b.clone();
    let mut d = a.clone() * b.clone() + b.clone().pow(3.0);
    c = c.clone() + (c.clone() + Value::from(1.0));
    c = c.clone() + (Value::from(1.0) + c.clone() + (-a.clone()));
    d = d.clone() + (d.clone() * Value::from(2.0) + (b.clone() + a.clone()).relu());
    d = d.clone() + (Value::from(3.0) * d.clone() + (b.clone() - a.clone()).relu());
    let e = c.clone() - d.clone();
    let f = e.pow(2.0);
    let mut g: Value<f64> = f.clone() / Value::from(2.0);
    g = g.clone() + (Value::from(10.0) / f.clone());
    g.backward();

    let tol = 0.00000001;

    assert!((g.val() - 24.70408163265306).abs() < tol);
    assert!((a.grad() - 138.83381924198252).abs() < tol);
    assert!((b.grad() - 645.5772594752187).abs() < tol);
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
