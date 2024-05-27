use std::ops::Mul;

use crate::{
    numeric::Numeric,
    shape::Shape,
    storage::{Layout, Storage},
    tensor2::Tensor2,
    vector2::Vector2,
};

pub const fn matrix_shape(m: usize, n: usize) -> Shape {
    [m, n, 0, 0, 0, 0]
}

#[derive(Tensor2, Debug, Clone)]
#[tensor_rank = 2]
#[tensor_shape = "matrix_shape(M, N)"]
pub struct Matrix2<T: Numeric, const M: usize, const N: usize> {
    pub(crate) storage: Storage<T>,
    pub layout: Layout,
}

impl<T: Numeric, const M: usize, const N: usize> Matrix2<T, M, N> {
    pub fn transpose(self) -> Matrix2<T, N, M> {
        Matrix2 {
            storage: self.storage,
            layout: self.layout.transpose(),
        }
    }
}

impl<'a, T: Numeric, const M: usize, const N: usize, const P: usize> Mul<&'a Matrix2<T, N, P>>
    for &'a Matrix2<T, M, N>
{
    type Output = Matrix2<T, M, P>;

    fn mul(self, other: &Matrix2<T, N, P>) -> Self::Output {
        matmul_impl::<T, M, N, P>(&self.storage, self.layout, &other.storage, other.layout)
    }
}

fn matmul_impl<T: Numeric, const M: usize, const N: usize, const P: usize>(
    a_storage: &[T],
    a_transpose: Layout,
    b_storage: &[T],
    b_transpose: Layout,
) -> Matrix2<T, M, P> {
    let mut out = Matrix2::zeros();
    // BLAS's output format is always column-major which is "transposed"
    // from our perspective
    out.layout = Layout::Transposed;

    unsafe {
        T::gemm(
            a_transpose.to_blas(),
            b_transpose.to_blas(),
            M as i32,
            P as i32,
            N as i32,
            T::one(),
            a_storage,
            if a_transpose.is_transposed() { M } else { N } as i32,
            b_storage,
            if b_transpose.is_transposed() { N } else { P } as i32,
            T::one(),
            &mut out.storage,
            M as i32,
        )
    }

    out
}

impl<'a, T: Numeric, const M: usize, const N: usize> Mul<&'a Vector2<T, N>>
    for &'a Matrix2<T, M, N>
{
    type Output = Vector2<T, M>;

    fn mul(self, other: &Vector2<T, N>) -> Self::Output {
        // BLAS always uses column-major format, so if we're "transposed" we're
        // already in BLAS format, otherwise we have to transpose.
        let mut out = Self::Output::zeros();
        let trans = self.layout.transpose().to_blas();
        let m = if self.layout.is_transposed() { M } else { N } as i32;
        let n = if self.layout.is_transposed() { N } else { M } as i32;
        let lda = m;

        unsafe {
            T::gemv(
                trans,
                m,
                n,
                T::one(),
                &self.storage,
                lda,
                &other.storage,
                1,
                T::one(),
                &mut out.storage,
                1,
            );
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_fn() {
        let x: Matrix2<f64, 2, 3> = Matrix2::from_fn(|_| 7.0);
        let y = x.map(|_, x| x * 2.0);
        assert_eq!(y[&[1, 1]], 14.0);
    }
}
