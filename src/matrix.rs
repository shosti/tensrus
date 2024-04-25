use crate::generic_tensor::GenericTensor;
use crate::numeric::Numeric;
use crate::scalar::Scalar;
use crate::tensor::{num_elems, IndexError, Tensor, TensorShape};
use num::ToPrimitive;
use std::ops::Mul;

pub const fn matrix_shape(m: usize, n: usize) -> TensorShape {
    [m, n, 0, 0, 0]
}

pub type Matrix<T, const M: usize, const N: usize> = MatrixTensor<T, 2, { matrix_shape(M, N) }>;

#[derive(Tensor, PartialEq, Debug)]
pub struct MatrixTensor<T: Numeric, const R: usize, const S: TensorShape>(GenericTensor<T, R, S>);

impl<T: Numeric, const R: usize, const S: TensorShape, F> From<[F; num_elems(R, S)]>
    for MatrixTensor<T, R, S>
where
    F: ToPrimitive,
{
    fn from(arr: [F; num_elems(R, S)]) -> Self {
        Self(GenericTensor::from(arr))
    }
}

impl<T: Numeric, const M: usize, const N: usize, F> From<[[F; N]; M]> for Matrix<T, M, N>
where
    [(); num_elems(2, matrix_shape(M, N))]:,
    F: ToPrimitive,
{
    fn from(arrs: [[F; N]; M]) -> Self {
        Self(arrs.into_iter().flatten().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::IndexError;

    #[test]
    #[rustfmt::skip]
    fn matrix_basics() {
        let x: Matrix<f64, _, _> = Matrix::from([
            [3, 4, 5],
            [2, 7, 9],
            [6, 5, 10],
            [3, 7, 3],
        ]);

        assert_eq!(x.shape(), [4, 3]);
        assert_eq!(x.get(&[2, 1]), Ok(5.0));
        assert_eq!(x.get(&[3, 2]), Ok(3.0));
        assert_eq!(x.get(&[4, 1]), Err(IndexError {}));

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
        let x: Matrix<f64, 3, 2> = [1, 2, 3].iter().cycle().map(|x| *x).collect();
        let y: Matrix<f64, _, _> = Matrix::from([
            [1.0, 2.0],
            [3.0, 1.0],
            [2.0, 3.0],
        ]);
        assert_eq!(x, y);
    }

    // #[test]
    // fn from_fn() {
    //     let x: Matrix<_, 3, 4> = Matrix::from_fn(|idx| {
    //         let [i, j] = idx;
    //         let s = format!("{}{}", i, j);
    //         s.parse().unwrap()
    //     });
    //     let y = Matrix::from([
    //         [0.0, 1.0, 2.0, 3.0],
    //         [10.0, 11.0, 12.0, 13.0],
    //         [20.0, 21.0, 22.0, 23.0],
    //     ]);

    //     assert_eq!(x, y);
    // }

    // #[test]
    // fn elem_mutiply() {
    //     let mut x = Matrix::from(
    //         [[2, 4, 6],
    //          [8, 10, 12]]
    //     );
    //     let y = Matrix::from(
    //         [[4, 8, 12],
    //          [16, 20, 24]]
    //     );

    //     x *= 2.0;
    //     assert_eq!(x, y);
    // }

    // #[test]
    // fn matrix_vector_conversions() {
    //     let x = Matrix::from(
    //         [[1, 2, 3],
    //          [4, 5, 6],
    //          [7, 8, 9],
    //          [10, 11, 12]]
    //     );

    //     assert_eq!(x.col(0), Ok(Vector::from([1, 4, 7, 10])));
    //     assert_eq!(x.col(1), Ok(Vector::from([2, 5, 8, 11])));
    //     assert_eq!(x.col(2), Ok(Vector::from([3, 6, 9, 12])));
    //     assert_eq!(x.col(3), Err(IndexError {}));
    //     assert_eq!(x.row(0), Ok(Vector::from([1, 2, 3])));
    //     assert_eq!(x.row(1), Ok(Vector::from([4, 5, 6])));
    //     assert_eq!(x.row(2), Ok(Vector::from([7, 8, 9])));
    //     assert_eq!(x.row(3), Ok(Vector::from([10, 11, 12])));
    //     assert_eq!(x.row(4), Err(IndexError {}));
    // }

    // #[test]
    // fn matrix_multiply() {
    //     let x = Matrix::from(
    //         [[1, 2],
    //          [3, 4],
    //          [5, 6]]
    //     );
    //     let y = Matrix::from(
    //         [[7, 8, 9, 10],
    //          [9, 10, 11, 12]]
    //     );
    //     let res = Matrix::from(
    //         [[25, 28, 31, 34],
    //          [57, 64, 71, 78],
    //          [89, 100, 111, 122]]
    //     );

    //     assert_eq!(x * y, res);
    // }

    // #[test]
    // fn matrix_vector_multiply() {
    //     let a = Matrix::from(
    //         [[1, -1, 2],
    //          [0, -3, 1]]
    //     );
    //     let x = Vector::from([2, 1, 0]);

    //     assert_eq!(a * x, Vector::from([1, -3]));
    // }

    // #[test]
    // fn transpose() {
    //     let x = Matrix::from(
    //         [[1, 2, 3],
    //          [4, 5, 6]]
    //     );
    //     let y = Matrix::from(
    //         [[1, 4],
    //          [2, 5],
    //          [3, 6]]
    //     );

    //     assert_eq!(x.transpose(), y);
    // }
}
