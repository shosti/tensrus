use proptest::prelude::*;
use seq_macro::seq;
#[cfg(feature = "slow_tests")]
use tensrus::shape::Transposable;
#[cfg(feature = "slow_tests")]
use tensrus::vector::Vector;
use tensrus::{matrix::Matrix, tensor::Tensor};

extern crate tensrus;

#[test]
fn test_matmul_distributive() {
    let a: Matrix<f64, 1, 1> = Matrix::zeros();
    let b: Matrix<f64, 1, 2> = Matrix::zeros();
    let c: Matrix<f64, 1, 2> = Matrix::zeros();

    assert_eq!(&a * &(b.clone() + &c), (&a * &b) + &(&a * &c));
}

seq!(N in 1..=20 {
    proptest! {
        #[test]
        #[cfg(feature = "slow_tests")]
        fn test_identity_~N(v in prop::collection::vec(any::<f64>(), N)) {
            let i = Matrix::<f64, N, N>::identity();
            let x: Matrix::<f64, N, N> = v.into_iter().collect();

            assert_eq!(&i * &x, x);
            assert_eq!(&x * &i, x);
        }
    }
});

#[cfg(feature = "slow_tests")]
fn assert_eq_within_tolerance<const M: usize, const N: usize>(
    a: Matrix<f64, M, N>,
    b: Matrix<f64, M, N>,
) {
    const TOLERANCE: f64 = 0.00001;
    for i in 0..M {
        for j in 0..N {
            assert!((a[&[i, j]] - b[&[i, j]]).abs() < TOLERANCE);
        }
    }
}

seq!(M in 1..=5 {
    seq!(N in 1..=5 {
        seq!(P in 1..=5 {
            proptest! {
                #[test]
                #[cfg(feature = "slow_tests")]
                #[allow(clippy::identity_op)]
                fn test_matmul_~M~N~P(v_a in proptest::collection::vec(-10000.0..10000.0, M * N),
                                      v_b in proptest::collection::vec(-10000.0..10000.0, N * P)) {
                    let a: Matrix::<f64, M, N> = v_a.into_iter().collect();
                    let b: Matrix::<f64, N, P> = v_b.into_iter().collect();

                    let c = &a * &b;

                    for i in 0..M {
                        for j in 0..P {
                            const TOLERANCE: f64 = 0.00001;
                            assert!((a.row(i).unwrap().to_owned().dot(&b.col(j).unwrap().to_owned()) - c[&[i, j]]).abs() < TOLERANCE);
                        }
                    }
                }

                #[test]
                #[cfg(feature = "slow_tests")]
                #[allow(clippy::identity_op)]
                fn test_matmul_distributivity_~M~N~P(v_a in proptest::collection::vec(-10000.0..10000.0, M * N),
                                                     v_b in proptest::collection::vec(-10000.0..10000.0, N * P),
                                                     v_c in proptest::collection::vec(-10000.0..10000.0, N * P)) {
                    let a: Matrix::<f64, M, N> = v_a.into_iter().collect();
                    let b: Matrix::<f64, N, P> = v_b.into_iter().collect();
                    let c: Matrix::<f64, N, P> = v_c.into_iter().collect();

                    assert_eq_within_tolerance(&a * &(b.clone() + &c), (&a * &b) + &(&a * &c));
                }

                #[test]
                #[cfg(feature = "slow_tests")]
                #[allow(clippy::identity_op)]
                fn test_matmul_transpose_~M~N~P(v_a in proptest::collection::vec(-10000.0..10000.0, M * N),
                                                v_b in proptest::collection::vec(-10000.0..10000.0, N * P)) {
                    let a: Matrix::<f64, M, N> = v_a.into_iter().collect();
                    let b: Matrix::<f64, N, P> = v_b.into_iter().collect();

                    assert_eq_within_tolerance(&a * &b, (&b.clone().transpose() * &a.clone().transpose()).transpose());
                }
            }
        });
    });
});

seq!(N in 1..=20 {
    proptest! {
        #[test]
        #[cfg(feature = "slow_tests")]
        fn test_matvecmul_identity_~N(v in prop::collection::vec(any::<f64>(), N)) {
            let i = Matrix::<f64, N, N>::identity();
            let x: Vector::<f64, N> = v.into_iter().collect();

            assert_eq!(&i * &x, x);
        }
    }
});

seq!(M in 1..=10 {
    seq!(N in 1..=10 {
        proptest! {
            #[test]
            #[cfg(feature = "slow_tests")]
            #[allow(clippy::identity_op)]
            fn test_matvecmul_~M~N(v_a in proptest::collection::vec(-10000.0..10000.0, M * N),
                                   v_x in proptest::collection::vec(-10000.0..10000.0, N)) {
                let a: Matrix::<f64, M, N> = v_a.into_iter().collect();
                let x: Vector::<f64, N> = v_x.into_iter().collect();

                let b = &a * &x;

                for i in 0..M {
                    const TOLERANCE: f64 = 0.00001;
                    assert!((a.row(i).unwrap().to_owned().dot(&x) - b[&[i]]).abs() < TOLERANCE);
                }
            }
        }
    });
});
