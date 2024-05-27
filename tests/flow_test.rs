#![feature(generic_arg_infer)]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
extern crate tensrus;

use std::fs::File;
use std::process::Command;
use tensrus::matrix2::Matrix2;
use tensrus::render::Graph;
use tensrus::scalar2::Scalar2;
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
    let mut g: Var<Scalar2<f64>> = f.clone() / Var::from(2.0);
    g = g.clone() + (Var::from(10.0) / f.clone());
    g.backward().unwrap();

    let tol = 0.00000001;

    assert!((g.data().val() - 24.70408163265306).abs() < tol);
    assert!((a.grad().unwrap().val() - 138.83381924198252).abs() < tol);
    assert!((b.grad().unwrap().val() - 645.5772594752187).abs() < tol);
}

#[test]
fn test_matmul() {
    let a = Var::new(Matrix2::<f64, _, _>::from([[7, 12], [42, 3], [8, 5]]));
    let b = Var::new(Matrix2::<f64, _, _>::from([[4, 17, 9], [11, 1, 15]]));
    let c = a.clone() * b.clone();
    let l = c.sum_elems();
    l.backward().unwrap();

    assert_eq!(l.data().val(), 2250.0);
    assert_eq!(
        a.grad().unwrap().clone(),
        Matrix2::<f64, _, _>::from([[30, 27], [30, 27], [30, 27]])
    );
    assert_eq!(
        b.grad().unwrap().clone(),
        Matrix2::<f64, _, _>::from([[57, 57, 57], [20, 20, 20]])
    );
}

fn render_graph(x: &Var<Scalar2<f64>>, id: String) {
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
