use cblas::{dgemm, dgemv, sgemm, sgemv, Layout, Transpose};

pub trait Numeric:
    num::Float
    + Copy
    + std::fmt::Display
    + std::fmt::Debug
    + std::ops::MulAssign
    + std::ops::AddAssign
    + rand::distributions::uniform::SampleUniform
    + BLASOps
    + 'static
{}

impl Numeric for f32 {}
impl Numeric for f64 {}

macro_rules! blas_ops {
    ( $( $name:ident ( $( $var:ident : $t:ty , )* ) , )* ) => {
        pub trait BLASOps: Sized {
            $( unsafe fn $name ( $( $var : $t , )* ) ; )*
        }

        impl BLASOps for f32 {
            $( unsafe fn $name ( $( $var : $t , )* ) {
                concat_idents!(s, $name) ( $( $var ),* )
            } )*
        }

        impl BLASOps for f64 {
            $( unsafe fn $name ( $( $var : $t , )* ) {
                concat_idents!(d, $name) ( $( $var ),* )
            } )*
        }
    };
}

blas_ops! {
    gemm(
        layout: Layout,
        transa: Transpose,
        transb: Transpose,
        m: i32,
        n: i32,
        k: i32,
        alpha: Self,
        a: &[Self],
        lda: i32,
        b: &[Self],
        ldb: i32,
        beta: Self,
        c: &mut [Self],
        ldc: i32,
    ),
    gemv(
        layout: Layout,
        transa: Transpose,
        m: i32,
        n: i32,
        alpha: Self,
        a: &[Self],
        lda: i32,
        x: &[Self],
        incx: i32,
        beta: Self,
        y: &mut [Self],
        incy: i32,
    ),
}
