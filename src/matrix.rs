use crate::generic_tensor::GenericTensor;
use crate::numeric::Numeric;
use crate::tensor::{num_elems, IndexError, ShapedTensor, TensorShape};
use crate::vector::{vector_shape, Vector};
use num::ToPrimitive;
use std::ops::Mul;

pub const fn matrix_shape(m: usize, n: usize) -> TensorShape {
    [m, n, 0, 0, 0]
}

#[derive(Tensor, PartialEq, Debug)]
#[tensor_rank = 2]
#[tensor_shape = "matrix_shape(M, N)"]
pub struct Matrix<T: Numeric, const M: usize, const N: usize>(
    GenericTensor<T, 2, { matrix_shape(M, N) }>,
)
where
    [(); num_elems(2, matrix_shape(M, N))]:;

impl<T: Numeric, const M: usize, const N: usize> Matrix<T, M, N>
where
    [(); num_elems(2, matrix_shape(M, N))]:,
{
    pub fn from_fn<F>(cb: F) -> Self
    where
        F: Fn([usize; 2]) -> T,
    {
        let t: GenericTensor<T, 2, { matrix_shape(M, N) }> = GenericTensor::from_fn(cb);
        Self(t)
    }

    pub fn row(&self, i: usize) -> Result<Vector<T, N>, IndexError>
    where
        [(); num_elems(1, vector_shape(N))]:,
    {
        if i >= M {
            return Err(IndexError {});
        }

        Ok(Vector::from_fn(|[j]| self.get([i, j])))
    }

    pub fn col(&self, j: usize) -> Result<Vector<T, M>, IndexError>
    where
        [(); num_elems(1, vector_shape(M))]:,
    {
        if j >= N {
            return Err(IndexError {});
        }

        Ok(Vector::from_fn(|[i]| self.get([i, j])))
    }

    // This could be "fast" but we'll deal with that later
    pub fn transpose(&self) -> Matrix<T, N, M>
    where
        [(); num_elems(2, matrix_shape(N, M))]:,
    {
        Matrix::from_fn(|[i, j]| self.get([j, i]))
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

impl<T: Numeric, const M: usize, const N: usize, const P: usize> Mul<Matrix<T, N, P>>
    for Matrix<T, M, N>
where
    [(); num_elems(2, matrix_shape(M, N))]:,
    [(); num_elems(2, matrix_shape(N, P))]:,
    [(); num_elems(2, matrix_shape(M, P))]:,
{
    type Output = Matrix<T, M, P>;

    fn mul(self, other: Matrix<T, N, P>) -> Self::Output {
        Self::Output::from_fn(|[i, j]| {
            let mut res = T::zero();
            for k in 0..N {
                res += self.get([i, k]) * other.get([k, j]);
            }
            res
        })
    }
}

impl<T: Numeric, const M: usize, const N: usize> Mul<Vector<T, N>> for Matrix<T, M, N>
where
    [(); num_elems(2, matrix_shape(M, N))]:,
    [(); num_elems(1, vector_shape(N))]:,
    [(); num_elems(1, vector_shape(M))]:,
{
    type Output = Vector<T, M>;

    fn mul(self, other: Vector<T, N>) -> Self::Output {
        Self::Output::from_fn(|[i]| {
            let mut res = T::zero();
            for k in 0..N {
                res += self.get([i, k]) * other.get([k]);
            }
            res
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::IndexError;
    use crate::tensor::ShapedTensor;
    use crate::vector::Vector;

    #[test]
    #[rustfmt::skip]
    fn matrix_basics() {
        let x: Matrix<f64, _, _> = Matrix::from([
            [3, 4, 5],
            [2, 7, 9],
            [6, 5, 10],
            [3, 7, 3],
        ]);

        assert_eq!(x.get([2, 1]), 5.0);
        assert_eq!(x.get([3, 2]), 3.0);

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
    fn from_iter() {
        let x: Matrix<f64, 3, 2> = [1, 2, 3].into_iter().cycle().collect();
        let y: Matrix<f64, _, _> = Matrix::from([
            [1.0, 2.0],
            [3.0, 1.0],
            [2.0, 3.0],
        ]);
        assert_eq!(x, y);
    }

    #[test]
    #[allow(clippy::zero_prefixed_literal)]
    fn from_fn() {
        let x: Matrix<f64, 3, 4> = Matrix::from_fn(|idx| {
            let [i, j] = idx;
            let s = format!("{}{}", i, j);
            s.parse().unwrap()
        });
        let y = Matrix::from([[00, 01, 02, 03], [10, 11, 12, 13], [20, 21, 22, 23]]);

        assert_eq!(x, y);
    }

    #[test]
    fn elem_mutiply() {
        let mut x = Matrix::from([[2, 4, 6], [8, 10, 12]]);
        let y = Matrix::from([[4, 8, 12], [16, 20, 24]]);

        x *= 2.0;
        assert_eq!(x, y);
    }

    #[test]
    fn matrix_vector_conversions() {
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
    fn matrix_multiply() {
        let x: Matrix<f64, _, _> = Matrix::from([[1, 2], [3, 4], [5, 6]]);
        let y: Matrix<f64, _, _> = Matrix::from([[7, 8, 9, 10], [9, 10, 11, 12]]);
        let res: Matrix<f64, _, _> =
            Matrix::from([[25, 28, 31, 34], [57, 64, 71, 78], [89, 100, 111, 122]]);

        assert_eq!(x * y, res);
    }

    #[test]
    fn matrix_vector_multiply() {
        let a: Matrix<f64, _, _> = Matrix::from([[1, -1, 2], [0, -3, 1]]);
        let x: Vector<f64, _> = Vector::from([2, 1, 0]);

        assert_eq!(a * x, Vector::from([1, -3]));
    }

    #[test]
    fn transpose() {
        let x: Matrix<f64, _, _> = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let y: Matrix<f64, _, _> = Matrix::from([[1, 4], [2, 5], [3, 6]]);

        assert_eq!(x.transpose(), y);
    }
}
