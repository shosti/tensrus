use crate::generic_tensor::GenericTensor;
use crate::numeric::Numeric;
use crate::slice::Slice;
use crate::tensor::{downrank, num_elems, IndexError, Shape, SlicedTensor, Tensor, Transpose};
use crate::vector::{vector_shape, Vector};
use num::ToPrimitive;
use std::ops::Mul;

pub const fn matrix_shape(m: usize, n: usize) -> Shape {
    [m, n, 0, 0, 0]
}

#[derive(Tensor, PartialEq, Debug)]
pub struct Matrix<T: Numeric, const M: usize, const N: usize>(
    GenericTensor<T, 2, { matrix_shape(M, N) }>,
)
where
    [(); num_elems(2, matrix_shape(M, N))]:;

impl<T: Numeric, const M: usize, const N: usize> Matrix<T, M, N>
where
    [(); num_elems(2, matrix_shape(M, N))]:,
{
    pub fn row(&self, i: usize) -> Result<Vector<T, N>, IndexError>
    where
        [(); num_elems(1, vector_shape(N))]:,
    {
        if i >= M {
            return Err(IndexError {});
        }

        Ok(Vector::from_fn(|[j]| self[&[i, *j]]))
    }

    pub fn col(&self, j: usize) -> Result<Vector<T, M>, IndexError>
    where
        [(); num_elems(1, vector_shape(M))]:,
    {
        if j >= N {
            return Err(IndexError {});
        }

        Ok(Vector::from_fn(|[i]| self[&[*i, j]]))
    }

    pub fn transpose(self) -> Matrix<T, N, M>
    where
        [(); num_elems(2, matrix_shape(N, M))]:,
    {
        Matrix(GenericTensor::new(
            self.0.storage,
            self.0.transpose.transpose(),
        ))
    }
}

impl<T: Numeric, const N: usize> Matrix<T, N, N>
where
    [(); num_elems(2, matrix_shape(N, N))]:,
{
    pub fn identity() -> Self {
        Self::from_fn(|[i, j]| if i == j { T::one() } else { T::zero() })
    }
}

impl<T: Numeric, const M: usize, const N: usize, F: ToPrimitive> From<[F; M * N]>
    for Matrix<T, M, N>
where
    [(); num_elems(2, matrix_shape(M, N))]:,
{
    fn from(arr: [F; M * N]) -> Self {
        let t: GenericTensor<T, 2, { matrix_shape(M, N) }> = arr.into_iter().collect();
        Self(t)
    }
}

impl<T: Numeric, const M: usize, const N: usize, F> From<[[F; N]; M]> for Matrix<T, M, N>
where
    [(); num_elems(2, matrix_shape(M, N))]:,
    F: ToPrimitive,
{
    fn from(arrs: [[F; N]; M]) -> Self {
        let t: GenericTensor<T, 2, { matrix_shape(M, N) }> = arrs.into_iter().flatten().collect();
        Self(t)
    }
}

impl<'a, T: Numeric, const M: usize, const N: usize, const P: usize> Mul<&'a Matrix<T, N, P>>
    for &'a Matrix<T, M, N>
where
    [(); num_elems(2, matrix_shape(M, N))]:,
    [(); num_elems(2, matrix_shape(N, P))]:,
    [(); num_elems(2, matrix_shape(M, P))]:,
{
    type Output = Matrix<T, M, P>;

    fn mul(self, other: &Matrix<T, N, P>) -> Self::Output {
        let mut out = Matrix::zeros();
        // BLAS's output format is always column-major which is "transposed"
        // from our perspective
        out.0.transpose = Transpose::Transposed;

        unsafe {
            T::gemm(
                self.0.transpose.to_blas(),
                other.0.transpose.to_blas(),
                M as i32,
                P as i32,
                N as i32,
                T::one(),
                &self.0.storage,
                if self.0.is_transposed() { M } else { N } as i32,
                &other.0.storage,
                if other.0.is_transposed() { N } else { P } as i32,
                T::one(),
                &mut out.0.storage,
                M as i32,
            )
        }
        out
    }
}

impl<'a, T: Numeric, const M: usize, const N: usize> Mul<&'a Vector<T, N>> for &'a Matrix<T, M, N>
where
    [(); num_elems(2, matrix_shape(M, N))]:,
    [(); num_elems(1, vector_shape(N))]:,
    [(); num_elems(1, vector_shape(M))]:,
{
    type Output = Vector<T, M>;

    fn mul(self, other: &Vector<T, N>) -> Self::Output {
        // BLAS always uses column-major format, so if we're "transposed" we're
        // already in BLAS format, otherwise we have to transpose.
        let mut out = Self::Output::zeros();
        let trans = if self.0.is_transposed() { b'N' } else { b'T' };
        let m = if self.0.is_transposed() { M } else { N } as i32;
        let n = if self.0.is_transposed() { N } else { M } as i32;
        let lda = m;

        unsafe {
            T::gemv(
                trans,
                m,
                n,
                T::one(),
                &self.0.storage,
                lda,
                &other.0.storage,
                1,
                T::one(),
                &mut out.0.storage,
                1,
            );
        }

        out
    }
}

impl<T: Numeric, const M: usize, const N: usize> SlicedTensor<T, 2, { matrix_shape(M, N) }>
    for Matrix<T, M, N>
where
    [(); num_elems(2, matrix_shape(M, N))]:,
{
    fn try_slice<const D: usize>(
        &self,
        idx: [usize; D],
    ) -> Result<Slice<T, { 2 - D }, { downrank(2, matrix_shape(M, N), D) }>, IndexError> {
        self.0.try_slice(idx)
    }
}

impl<'a, T: Numeric, const M: usize, const N: usize> From<Slice<'a, T, 2, { matrix_shape(M, N) }>>
    for Matrix<T, M, N>
where
    [(); num_elems(2, matrix_shape(M, N))]:,
{
    fn from(s: Slice<'a, T, 2, { matrix_shape(M, N) }>) -> Self {
        let t = s.into();
        Self(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::IndexError;
    use crate::vector::Vector;
    use proptest::prelude::*;
    use seq_macro::seq;

    #[test]
    #[rustfmt::skip]
    fn test_matrix_basics() {
        let x: Matrix<f64, _, _> = Matrix::from([
            [3, 4, 5],
            [2, 7, 9],
            [6, 5, 10],
            [3, 7, 3],
        ]);

        assert_eq!(x[&[2, 1]], 5.0);
        assert_eq!(x[&[3, 2]], 3.0);

        let y: Matrix<f64, 4, 3> = Matrix::from([
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
        let x: Matrix<f64, 3, 2> = [1, 2, 3].into_iter().cycle().collect();
        let y: Matrix<f64, _, _> = Matrix::from([
            [1.0, 2.0],
            [3.0, 1.0],
            [2.0, 3.0],
        ]);
        assert_eq!(x, y);
    }

    seq!(N in 1..=20 {
        proptest! {
            #[test]
            #[cfg(feature = "proptest")]
            fn test_identity_~N(v in prop::collection::vec(any::<f64>(), N)) {
                let i = Matrix::<f64, N, N>::identity();
                let x: Matrix::<f64, N, N> = v.into_iter().collect();

                assert_eq!(&i * &x, x);
                assert_eq!(&x * &i, x);
            }
        }
    });

    seq!(N in 1..=10 {
        seq!(M in 1..=10 {
            seq!(P in 1..10 {
                proptest! {
                    #[test]
                    #[cfg(feature = "proptest")]
                    fn test_matmul_~N~M~P(v_a in proptest::collection::vec((-10000.0)..(10000.0), N * M),
                                          v_b in proptest::collection::vec((-10000.0)..(10000.0), M * P)) {
                        let a: Matrix::<f64, N, M> = v_a.into_iter().collect();
                        let b: Matrix::<f64, M, P> = v_b.into_iter().collect();

                        let c = &a * &b;

                        for i in 0..N {
                            for j in 0..P {
                                const TOLERANCE: f64 = 0.00001;
                                assert!((a.row(i).unwrap().dot(&b.col(j).unwrap()) - c[&[i, j]]).abs() < TOLERANCE);
                            }
                        }
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
                #[cfg(feature = "proptest")]
                fn test_matvecmul_~M~N(v_a in proptest::collection::vec((-10000.0)..(10000.0), M * N),
                                       v_x in proptest::collection::vec((-10000.0)..(10000.0), N)) {
                    let a: Matrix::<f64, M, N> = v_a.into_iter().collect();
                    let x: Vector::<f64, N> = v_x.into_iter().collect();

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
        let x: Matrix<f64, 3, 4> = Matrix::from_fn(|idx| {
            let [i, j] = idx;
            let s = format!("{}{}", i, j);
            s.parse().unwrap()
        });
        let y = Matrix::from([[00, 01, 02, 03], [10, 11, 12, 13], [20, 21, 22, 23]]);

        assert_eq!(x, y);
    }

    #[test]
    fn test_matrix_vector_conversions() {
        let x: Matrix<f64, _, _> = Matrix::from([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);

        assert_eq!(x.col(0), Ok(Vector::from([1, 4, 7, 10])));
        assert_eq!(x.col(1), Ok(Vector::from([2, 5, 8, 11])));
        assert_eq!(x.col(2), Ok(Vector::from([3, 6, 9, 12])));
        assert_eq!(x.col(3), Err(IndexError {}));
        assert_eq!(x.row(0), Ok(Vector::from([1, 2, 3])));
        assert_eq!(x.row(1), Ok(Vector::from([4, 5, 6])));
        assert_eq!(x.row(2), Ok(Vector::from([7, 8, 9])));
        assert_eq!(x.row(3), Ok(Vector::from([10, 11, 12])));
        assert_eq!(x.row(4), Err(IndexError {}));
    }

    #[test]
    fn test_matrix_multiply() {
        #[rustfmt::skip]
        let x: Matrix<f64, _, _> = Matrix::from([
            [1, 2],
            [3, 4],
            [5, 6]
        ]);
        #[rustfmt::skip]
        let x_t: Matrix<f64, _, _> = Matrix::from([
            [1, 3, 5],
            [2, 4, 6],
        ]).transpose();

        #[rustfmt::skip]
        let y: Matrix<f64, _, _> = Matrix::from([
            [7, 8, 9, 10],
            [9, 10, 11, 12]
        ]);
        #[rustfmt::skip]
        let y_t: Matrix<f64, _, _> = Matrix::from([
            [7, 9],
            [8, 10],
            [9, 11],
            [10, 12],
        ]).transpose();

        #[rustfmt::skip]
        let want: Matrix<f64, _, _> = Matrix::from([
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
        let a: Matrix<f64, _, _> = Matrix::from([
            [1, -1, 2],
            [0, -3, 1],
        ]);
        #[rustfmt::skip]
        let a_t: Matrix<f64, _, _> = Matrix::from([
            [1, 0],
            [-1, -3],
            [2, 1],
        ]).transpose();
        let x: Vector<f64, _> = Vector::from([2, 1, 0]);

        let want = Vector::from([1, -3]);

        assert_eq!(&a * &x, want);
        assert_eq!(&a_t * &x, want);
    }

    #[test]
    fn test_transpose() {
        let x: Matrix<f64, _, _> = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let y: Matrix<f64, _, _> = Matrix::from([[1, 4], [2, 5], [3, 6]]);

        assert_eq!(x.transpose(), y);
    }
}
