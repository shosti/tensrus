extern crate tensrus;

use std::fs::File;
use std::process::Command;
use tensrus::render::Graph;
use tensrus::scalar::Scalar;
use tensrus::var::Var;

#[test]
fn flow_sanity_test() {
    let x = Var::from(-4.0);
    let z = Var::from(2.0) * x.clone() + Var::from(2.0) + x.clone();
    let q = z.clone().relu() + z.clone() * x.clone();
    let h = (z.clone() * z.clone()).relu();
    let y = h.clone() + q.clone() + q.clone() * x.clone();
    y.backward().unwrap();
    render_graph(&y, "thing".to_string());

    assert_eq!(y.data().val(), -20.0);
    assert_eq!(x.grad().unwrap().val(), 46.0);
}

#[test]
fn test_more_ops() {
    let a = Var::from(-4.0);
    let b = Var::from(2.0);
    let mut c = a.clone() + b.clone();
    let mut d = a.clone() * b.clone() + b.clone().elem_pow(3.0);
    c = c.clone() + (c.clone() + Var::from(1.0));
    c = c.clone() + (Var::from(1.0) + c.clone() + (-a.clone()));
    d = d.clone() + (d.clone() * Var::from(2.0) + (b.clone() + a.clone()).relu());
    d = d.clone() + (Var::from(3.0) * d.clone() + (b.clone() - a.clone()).relu());
    let e = c.clone() - d.clone();
    let f = e.elem_pow(2.0);
    let mut g: Var<Scalar<f64>> = f.clone() / Var::from(2.0);
    g = g.clone() + (Var::from(10.0) / f.clone());
    g.backward().unwrap();

    let tol = 0.00000001;

    assert!((g.data().val() - 24.70408163265306).abs() < tol);
    assert!((a.grad().unwrap().val() - 138.83381924198252).abs() < tol);
    assert!((b.grad().unwrap().val() - 645.5772594752187).abs() < tol);
}

fn render_graph(x: &Var<Scalar<f64>>, id: String) {
    let g = Graph::new(x);
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
