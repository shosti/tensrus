use std::ops::Mul;

use num::ToPrimitive;

use crate::{
    numeric::Numeric,
    shape::Shape,
    storage::{num_elems, IndexError, Layout, Storage},
    tensor2::Tensor2,
    type_assert::{Assert, IsTrue},
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
    pub fn row(&self, i: usize) -> Result<Vector2<T, N>, IndexError> {
        if i >= M {
            return Err(IndexError {});
        }

        Ok(Vector2::from_fn(|[j]| self[&[i, *j]]))
    }

    pub fn col(&self, j: usize) -> Result<Vector2<T, M>, IndexError> {
        if j >= N {
            return Err(IndexError {});
        }

        Ok(Vector2::from_fn(|[i]| self[&[*i, j]]))
    }

    pub fn view(&self) -> MatrixView<T, M, N> {
        MatrixView {
            storage: &self.storage,
            layout: self.layout,
        }
    }

    pub fn transpose(self) -> Matrix2<T, N, M> {
        Matrix2 {
            storage: self.storage,
            layout: self.layout.transpose(),
        }
    }
}

impl<T: Numeric, const N: usize> Matrix2<T, N, N> {
    pub fn identity() -> Self {
        Self::from_fn(|[i, j]| if i == j { T::one() } else { T::zero() })
    }
}

impl<T: Numeric, const M: usize, const N: usize, U> From<[[U; N]; M]> for Matrix2<T, M, N>
where
    U: ToPrimitive,
{
    fn from(arrs: [[U; N]; M]) -> Self {
        arrs.into_iter().flatten().collect()
    }
}

impl<T: Numeric, const M: usize, const N: usize, F: ToPrimitive> From<[F; M * N]>
    for Matrix2<T, M, N>
{
    fn from(arr: [F; M * N]) -> Self {
        arr.into_iter().collect()
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

pub struct MatrixView<'a, T: Numeric, const M: usize, const N: usize> {
    storage: &'a [T],
    layout: Layout,
}

impl<'a, T: Numeric, const M: usize, const N: usize> MatrixView<'a, T, M, N> {
    pub fn transpose(&self) -> MatrixView<'a, T, N, M> {
        MatrixView {
            storage: self.storage,
            layout: self.layout.transpose(),
        }
    }

    pub fn reshape<const M2: usize, const N2: usize>(&self) -> MatrixView<T, M2, N2>
    where
        Assert<{ num_elems(2, matrix_shape(M, N)) == num_elems(2, matrix_shape(M2, N2)) }>: IsTrue,
    {
        MatrixView {
            storage: self.storage,
            layout: self.layout,
        }
    }
}

impl<'a, T: Numeric, const M: usize, const N: usize> Clone for MatrixView<'a, T, M, N> {
    fn clone(&self) -> Self {
        MatrixView {
            storage: self.storage,
            layout: self.layout,
        }
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
        let trans = self.layout.to_blas();
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

impl<'a, T: Numeric, const M: usize, const N: usize, const P: usize> Mul<MatrixView<'a, T, N, P>>
    for &'a Matrix2<T, M, N>
{
    type Output = Matrix2<T, M, P>;

    fn mul(self, other: MatrixView<'a, T, N, P>) -> Self::Output {
        matmul_impl::<T, M, N, P>(&self.storage, self.layout, other.storage, other.layout)
    }
}

impl<'a, T: Numeric, const M: usize, const N: usize, const P: usize> Mul<&'a Matrix2<T, N, P>>
    for MatrixView<'a, T, M, N>
{
    type Output = Matrix2<T, M, P>;

    fn mul(self, other: &'a Matrix2<T, N, P>) -> Self::Output {
        matmul_impl::<T, M, N, P>(self.storage, self.layout, &other.storage, other.layout)
    }
}

impl<'a, T: Numeric, const M: usize, const N: usize, const P: usize> Mul<MatrixView<'a, T, N, P>>
    for MatrixView<'a, T, M, N>
{
    type Output = Matrix2<T, M, P>;

    fn mul(self, other: MatrixView<'a, T, N, P>) -> Self::Output {
        matmul_impl::<T, M, N, P>(self.storage, self.layout, other.storage, other.layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector2::Vector2;
    use proptest::prelude::*;
    use seq_macro::seq;

    #[test]
    #[rustfmt::skip]
    fn test_matrix_basics() {
        let x: Matrix2<f64, _, _> = Matrix2::from([
            [3, 4, 5],
            [2, 7, 9],
            [6, 5, 10],
            [3, 7, 3],
        ]);

        assert_eq!(x[&[2, 1]], 5.0);
        assert_eq!(x[&[3, 2]], 3.0);

        let y: Matrix2<f64, 4, 3> = Matrix2::from([
            3.0, 4.0, 5.0,
            2.0, 7.0, 9.0,
            6.0, 5.0, 10.0,
            3.0, 7.0, 3.0
        ]);
        assert_eq!(x, y);
    }

    #[test]
    #[rustfmt::skip]
    fn test_from_iter() {
        let x: Matrix2<f64, 3, 2> = [1, 2, 3].into_iter().cycle().collect();
        let y: Matrix2<f64, _, _> = Matrix2::from([
            [1.0, 2.0],
            [3.0, 1.0],
            [2.0, 3.0],
        ]);
        assert_eq!(x, y);
    }

    #[test]
    fn test_matmul_distributive() {
        let a: Matrix2<f64, 1, 1> = Matrix2::zeros();
        let b: Matrix2<f64, 1, 2> = Matrix2::zeros();
        let c: Matrix2<f64, 1, 2> = Matrix2::zeros();

        assert_eq!(&a * &(b.clone() + &c), (&a * &b) + &(&a * &c));
    }

    seq!(N in 1..=20 {
        proptest! {
            #[test]
            #[cfg(feature = "proptest")]
            fn test_identity_~N(v in prop::collection::vec(any::<f64>(), N)) {
                let i = Matrix2::<f64, N, N>::identity();
                let x: Matrix2::<f64, N, N> = v.into_iter().collect();

                assert_eq!(&i * &x, x);
                assert_eq!(&x * &i, x);
            }
        }
    });

    #[cfg(feature = "proptest")]
    fn assert_eq_within_tolerance<const M: usize, const N: usize>(
        a: Matrix2<f64, M, N>,
        b: Matrix2<f64, M, N>,
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
                    #[cfg(feature = "proptest")]
                    #[allow(clippy::identity_op)]
                    fn test_matmul_~M~N~P(v_a in proptest::collection::vec(-10000.0..10000.0, M * N),
                                          v_b in proptest::collection::vec(-10000.0..10000.0, N * P)) {
                        let a: Matrix2::<f64, M, N> = v_a.into_iter().collect();
                        let b: Matrix2::<f64, N, P> = v_b.into_iter().collect();

                        let c = &a * &b;

                        for i in 0..M {
                            for j in 0..P {
                                const TOLERANCE: f64 = 0.00001;
                                assert!((a.row(i).unwrap().dot(&b.col(j).unwrap()) - c[&[i, j]]).abs() < TOLERANCE);
                            }
                        }
                    }

                    #[test]
                    #[cfg(feature = "proptest")]
                    #[allow(clippy::identity_op)]
                    fn test_matmul_distributivity_~M~N~P(v_a in proptest::collection::vec(-10000.0..10000.0, M * N),
                                                         v_b in proptest::collection::vec(-10000.0..10000.0, N * P),
                                                         v_c in proptest::collection::vec(-10000.0..10000.0, N * P)) {
                        let a: Matrix2::<f64, M, N> = v_a.into_iter().collect();
                        let b: Matrix2::<f64, N, P> = v_b.into_iter().collect();
                        let c: Matrix2::<f64, N, P> = v_c.into_iter().collect();

                        assert_eq_within_tolerance(&a * &(b.clone() + &c), (&a * &b) + &(&a * &c));
                    }

                    #[test]
                    #[cfg(feature = "proptest")]
                    #[allow(clippy::identity_op)]
                    fn test_matmul_transpose_~M~N~P(v_a in proptest::collection::vec(-10000.0..10000.0, M * N),
                                                    v_b in proptest::collection::vec(-10000.0..10000.0, N * P)) {
                        let a: Matrix2::<f64, M, N> = v_a.into_iter().collect();
                        let b: Matrix2::<f64, N, P> = v_b.into_iter().collect();

                        assert_eq_within_tolerance(&a * &b, (&b.clone().transpose() * &a.clone().transpose()).transpose());
                        assert_eq_within_tolerance(&a * b.view(), (&b.clone().transpose() * a.view().transpose()).transpose());
                        assert_eq_within_tolerance(a.view() * &b, (&b.clone().transpose() * a.view().transpose()).transpose());
                        assert_eq_within_tolerance(a.view() * b.view(), (b.view().transpose() * a.view().transpose()).transpose());
                    }
                }
            });
        });
    });

    seq!(N in 1..=20 {
        proptest! {
            #[test]
            #[cfg(feature = "proptest")]
            fn test_matvecmul_identity_~N(v in prop::collection::vec(any::<f64>(), N)) {
                let i = Matrix2::<f64, N, N>::identity();
                let x: Vector2::<f64, N> = v.into_iter().collect();

                assert_eq!(&i * &x, x);
            }
        }
    });

    seq!(M in 1..=10 {
        seq!(N in 1..=10 {
            proptest! {
                #[test]
                #[cfg(feature = "proptest")]
                #[allow(clippy::identity_op)]
                fn test_matvecmul_~M~N(v_a in proptest::collection::vec(-10000.0..10000.0, M * N),
                                       v_x in proptest::collection::vec(-10000.0..10000.0, N)) {
                    let a: Matrix2::<f64, M, N> = v_a.into_iter().collect();
                    let x: Vector2::<f64, N> = v_x.into_iter().collect();

                    let b = &a * &x;

                    for i in 0..M {
                        const TOLERANCE: f64 = 0.00001;
                        assert!((a.row(i).unwrap().dot(&x) - b[&[i]]).abs() < TOLERANCE);
                    }
                }
            }
        });
    });

    #[test]
    #[allow(clippy::zero_prefixed_literal)]
    fn test_from_fn() {
        let x: Matrix2<f64, 3, 4> = Matrix2::from_fn(|idx| {
            let [i, j] = idx;
            let s = format!("{}{}", i, j);
            s.parse().unwrap()
        });
        let y = Matrix2::from([[00, 01, 02, 03], [10, 11, 12, 13], [20, 21, 22, 23]]);

        assert_eq!(x, y);
    }

    #[test]
    fn test_matrix_vector_conversions() {
        let x: Matrix2<f64, _, _> = Matrix2::from([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);

        assert_eq!(x.col(0), Ok(Vector2::from([1, 4, 7, 10])));
        assert_eq!(x.col(1), Ok(Vector2::from([2, 5, 8, 11])));
        assert_eq!(x.col(2), Ok(Vector2::from([3, 6, 9, 12])));
        assert_eq!(x.col(3), Err(IndexError {}));
        assert_eq!(x.row(0), Ok(Vector2::from([1, 2, 3])));
        assert_eq!(x.row(1), Ok(Vector2::from([4, 5, 6])));
        assert_eq!(x.row(2), Ok(Vector2::from([7, 8, 9])));
        assert_eq!(x.row(3), Ok(Vector2::from([10, 11, 12])));
        assert_eq!(x.row(4), Err(IndexError {}));
    }

    #[test]
    fn test_matrix_multiply() {
        #[rustfmt::skip]
        let x: Matrix2<f64, _, _> = Matrix2::from([
            [1, 2],
            [3, 4],
            [5, 6]
        ]);
        #[rustfmt::skip]
        let x_t: Matrix2<f64, _, _> = Matrix2::from([
            [1, 3, 5],
            [2, 4, 6],
        ]).transpose();

        #[rustfmt::skip]
        let y: Matrix2<f64, _, _> = Matrix2::from([
            [7, 8, 9, 10],
            [9, 10, 11, 12]
        ]);
        #[rustfmt::skip]
        let y_t: Matrix2<f64, _, _> = Matrix2::from([
            [7, 9],
            [8, 10],
            [9, 11],
            [10, 12],
        ]).transpose();

        #[rustfmt::skip]
        let want: Matrix2<f64, _, _> = Matrix2::from([
            [25, 28, 31, 34],
            [57, 64, 71, 78],
            [89, 100, 111, 122]
        ]);

        // assert_eq!(&x * &y, want);
        assert_eq!(&x_t * &y, want);
        assert_eq!(&x * &y_t, want);
        assert_eq!(&x_t * &y_t, want);
    }

    #[test]
    fn test_matrix_vector_multiply() {
        #[rustfmt::skip]
        let a: Matrix2<f64, _, _> = Matrix2::from([
            [1, -1, 2],
            [0, -3, 1],
        ]);
        #[rustfmt::skip]
        let a_t: Matrix2<f64, _, _> = Matrix2::from([
            [1, 0],
            [-1, -3],
            [2, 1],
        ]).transpose();
        let x: Vector2<f64, _> = Vector2::from([2, 1, 0]);

        let want = Vector2::from([1, -3]);

        assert_eq!(&a * &x, want);
        assert_eq!(&a_t * &x, want);
    }

    #[test]
    fn test_transpose() {
        let x: Matrix2<f64, _, _> = Matrix2::from([[1, 2, 3], [4, 5, 6]]);
        let y: Matrix2<f64, _, _> = Matrix2::from([[1, 4], [2, 5], [3, 6]]);

        assert_eq!(x.transpose(), y);
    }
}
