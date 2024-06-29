use blas::{ddot, dgemm, dgemv, sdot, sgemm, sgemv};

macro_rules! blas_ops {
    ( $( $name:ident ( $( $var:ident : $t:ty , )* ) -> $ret:ty , )* ) => {
        #[allow(clippy::missing_safety_doc)]
        #[allow(clippy::too_many_arguments)]
        pub trait BLASOps: Sized {
            $(
                unsafe fn $name ( $( $var : $t , )* ) -> $ret ;
            )*
        }

        impl BLASOps for f32 {
            $( unsafe fn $name ( $( $var : $t , )* )  -> $ret {
                concat_idents!(s, $name) ( $( $var ),* )
            } )*
        }

        impl BLASOps for f64 {
            $( unsafe fn $name ( $( $var : $t , )* ) -> $ret {
                concat_idents!(d, $name) ( $( $var ),* )
            } )*
        }
    };
}

blas_ops! {
    gemm(
        transa: u8,
        transb: u8,
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
    ) -> (),
    gemv(
        trans: u8,
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
    ) -> (),
    dot(
        n: i32,
        x: &[Self],
        incx: i32,
        y: &[Self],
        incy: i32,
    ) -> Self,
}
