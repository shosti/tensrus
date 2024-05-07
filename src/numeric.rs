use cblas::{dgemm, sgemm, Layout, Transpose};

pub trait Numeric:
    num::Float
    + Copy
    + std::fmt::Display
    + std::fmt::Debug
    + std::ops::MulAssign
    + std::ops::AddAssign
    + rand::distributions::uniform::SampleUniform
    + 'static
{
    unsafe fn gemm(
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
    );
}

impl Numeric for f32 {
    unsafe fn gemm(
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
    ) {
        sgemm(
            layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        )
    }
}
impl Numeric for f64 {
    unsafe fn gemm(
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
    ) {
        dgemm(
            layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        )
    }
}
