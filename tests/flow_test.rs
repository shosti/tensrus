#![feature(generic_arg_infer)]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
extern crate tensrus;

use tensrus::matrix::Matrix;
use tensrus::scalar::Scalar;
use tensrus::var::Var;

#[test]
fn flow_sanity_test() {
    let x = Var::param(Scalar::from(-4.0));
    let z = Var::param(Scalar::from(2.0)) * x.clone() + Var::param(Scalar::from(2.0)) + x.clone();
    let q = z.clone().relu() + z.clone() * x.clone();
    let h = (z.clone() * z.clone()).relu();
    let y = h.clone() + q.clone() + q.clone() * x.clone();
    y.backward().unwrap();

    assert_eq!(y.data().val(), -20.0);
    assert_eq!(x.as_param().unwrap().grad().unwrap().val(), 46.0);
}

#[test]
fn test_more_ops() {
    let a = Var::param(Scalar::from(-4.0));
    let b = Var::param(Scalar::from(2.0));
    let mut c = a.clone() + b.clone();
    let mut d = a.clone() * b.clone() + b.clone().elem_pow(3.0);
    c = c.clone() + (c.clone() + Var::param(Scalar::from(1.0)));
    c = c.clone() + (Var::param(Scalar::from(1.0)) + c.clone() + (-a.clone()));
    d = d.clone() + (d.clone() * Var::param(Scalar::from(2.0)) + (b.clone() + a.clone()).relu());
    d = d.clone() + (Var::param(Scalar::from(3.0)) * d.clone() + (b.clone() - a.clone()).relu());
    let e = c.clone() - d.clone();
    let f = e.elem_pow(2.0);
    let mut g: Var<Scalar<f64>> = f.clone() / Var::param(Scalar::from(2.0));
    g = g.clone() + (Var::param(Scalar::from(10.0)) / f.clone());
    g.backward().unwrap();

    let tol = 0.00000001;

    assert!((g.data().val() - 24.70408163265306).abs() < tol);
    assert!((a.as_param().unwrap().grad().unwrap().val() - 138.83381924198252).abs() < tol);
    assert!((b.as_param().unwrap().grad().unwrap().val() - 645.5772594752187).abs() < tol);
}

#[test]
fn test_matmul() {
    let a = Var::param(Matrix::<f64, _, _>::from([
        [7.0, 12.0],
        [42.0, 3.0],
        [8.0, 5.0],
    ]));
    let b = Var::param(Matrix::<f64, _, _>::from([
        [4.0, 17.0, 9.0],
        [11.0, 1.0, 15.0],
    ]));
    let c = a.clone() * b.clone();
    let l = c.sum_elems();
    l.backward().unwrap();

    assert_eq!(l.data().val(), 2250.0);
    assert_eq!(
        a.as_param().unwrap().grad().unwrap().clone(),
        Matrix::<f64, _, _>::from([[30.0, 27.0], [30.0, 27.0], [30.0, 27.0]])
    );
    assert_eq!(
        b.as_param().unwrap().grad().unwrap().clone(),
        Matrix::<f64, _, _>::from([[57.0, 57.0, 57.0], [20.0, 20.0, 20.0]])
    );
}
