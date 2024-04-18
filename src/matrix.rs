use crate::numeric::Numeric;
use crate::tensor::{IndexError, Tensor};
use crate::vector::Vector;
use std::cell::RefCell;
use std::ops::{Mul, MulAssign};
use std::rc::Rc;

pub struct Matrix<T: Numeric, const M: usize, const N: usize> {
    vals: Rc<RefCell<Vec<T>>>,
    transposed: bool,
}

impl<T: Numeric, const M: usize, const N: usize> Matrix<T, M, N> {
    pub fn col(&self, j: usize) -> Result<Vector<T, M>, IndexError> {
        if j >= N {
            return Err(IndexError {});
        }

        Ok(Vector::from_fn(|idx| self.get([idx[0], j]).unwrap()))
    }

    pub fn row(&self, i: usize) -> Result<Vector<T, N>, IndexError> {
        if i >= M {
            return Err(IndexError {});
        }

        Ok(Vector::from_fn(|idx| self.get([i, idx[0]]).unwrap()))
    }

    fn in_bounds(&self, i: usize, j: usize) -> bool {
        i < M && j < N
    }

    fn idx(&self, i: usize, j: usize) -> usize {
        if self.transposed {
            (j * M) + i
        } else {
            (i * N) + j
        }
    }
}

impl<T: Numeric, const M: usize, const N: usize> From<[[T; N]; M]> for Matrix<T, M, N> {
    fn from(arrs: [[T; N]; M]) -> Self {
        let mut vals: Vec<T> = Vec::with_capacity(M * N);
        for idx in 0..(M * N) {
            let i = idx / N;
            let j = idx % N;

            vals.push(arrs[i][j]);
        }

        Self {
            vals: Rc::new(RefCell::new(vals)),
            transposed: false,
        }
    }
}

impl<T: Numeric, const M: usize, const N: usize> Tensor<T, 2> for Matrix<T, M, N> {
    type Transpose = Matrix<T, N, M>;

    fn from_fn<F>(mut cb: F) -> Self
    where
        F: FnMut([usize; 2]) -> T,
    {
        let vals = Rc::new(RefCell::new(
            (0..(M * N)).map(|idx| cb([idx / N, idx % N])).collect(),
        ));
        Self {
            vals,
            transposed: false,
        }
    }

    fn shape(&self) -> [usize; 2] {
        [M, N]
    }

    fn get(&self, idx: [usize; 2]) -> Result<T, IndexError> {
        let [i, j] = idx;
        if !self.in_bounds(i, j) {
            return Err(IndexError {});
        }

        Ok(self.vals.borrow()[self.idx(i, j)])
    }

    fn set(&mut self, idx: [usize; 2], val: T) -> Result<(), IndexError> {
        let [i, j] = idx;
        if !self.in_bounds(i, j) {
            return Err(IndexError {});
        }
        let idx = self.idx(i, j);

        self.vals.borrow_mut()[idx] = val;

        Ok(())
    }

    fn transpose(&self) -> Self::Transpose {
        Self::Transpose {
            vals: self.vals.clone(),
            transposed: true,
        }
    }
}

impl<T: Numeric, const M: usize, const N: usize> MulAssign<T> for Matrix<T, M, N> {
    fn mul_assign(&mut self, other: T) {
        self.vals
            .borrow_mut()
            .iter_mut()
            .for_each(|n| *n *= other);
    }
}

impl<T: Numeric, const M: usize, const N: usize> PartialEq for Matrix<T, M, N> {
    fn eq(&self, other: &Self) -> bool {
        for i in 0..M {
            for j in 0..N {
                if self.get([i, j]).unwrap() != other.get([i, j]).unwrap() {
                    return false;
                }
            }
        }

        true
    }
}

impl<T: Numeric, const M: usize, const N: usize, const P: usize> Mul<Matrix<T, N, P>>
    for Matrix<T, M, N>
{
    type Output = Matrix<T, M, P>;

    fn mul(self, other: Matrix<T, N, P>) -> Self::Output {
        Matrix::from_fn(|idx| {
            let [i, j] = idx;
            self.row(i).unwrap().dot(&other.col(j).unwrap())
        })
    }
}

impl<T: Numeric, const M: usize, const N: usize> Mul<Vector<T, N>> for Matrix<T, M, N> {
    type Output = Vector<T, M>;

    fn mul(self, other: Vector<T, N>) -> Self::Output {
        Vector::from_fn(|idx| self.row(idx[0]).unwrap().dot(&other))
    }
}

impl<T: Numeric, const M: usize, const N: usize> Eq for Matrix<T, M, N> {}

impl<T: Numeric, const M: usize, const N: usize> std::fmt::Debug for Matrix<T, M, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut repr = String::from(format!("{}x{} Matrix", M, N));
        repr.push_str(" {\n [");
        for i in 0..M {
            for j in 0..N {
                repr.push_str(&format!("{} ", self.get([i, j]).unwrap()));
            }
            repr.push_str("\n");
        }
        repr.push_str("]\n");

        write!(f, "{}", repr)
    }
}

#[cfg(test)]
#[rustfmt::skip]
mod tests {
    use super::*;

    #[test]
    fn basics() {
        let x = Matrix::from(
            [[3, 4, 5],
             [2, 7, 9],
             [6, 5, 10],
             [3, 7, 3]]
        );

        assert_eq!(x.shape(), [4, 3]);
        assert_eq!(x.get([2, 1]), Ok(5));
        assert_eq!(x.get([3, 2]), Ok(3));
        assert_eq!(x.get([4, 1]), Err(IndexError {}));
    }

    #[test]
    fn from_fn() {
        let x: Matrix<_, 3, 4> = Matrix::from_fn(|idx| {
            let [i, j] = idx;
            let s = format!("{}{}", i, j);
            s.parse().unwrap()
        });
        let y = Matrix::from(
            [[00, 01, 02, 03],
             [10, 11, 12, 13],
             [20, 21, 22, 23]]
        );

        assert_eq!(x, y);
    }

    #[test]
    fn equality() {
        let x = Matrix::from(
            [[1, 2, 3],
             [4, 5, 6]]
        );
        let y = Matrix::from(
            [[1, 2, 3],
             [4, 5, 6]]
        );

        assert_eq!(x, y);
    }

    #[test]
    fn elem_mutiply() {
        let mut x = Matrix::from(
            [[2, 4, 6],
             [8, 10, 12]]
        );
        let y = Matrix::from(
            [[4, 8, 12],
             [16, 20, 24]]
        );

        x *= 2;
        assert_eq!(x, y);
    }

    #[test]
    fn matrix_vector_conversions() {
        let x = Matrix::from(
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9],
             [10, 11, 12]]
        );

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
        let x = Matrix::from(
            [[1, 2],
             [3, 4],
             [5, 6]]
        );
        let y = Matrix::from(
            [[7, 8, 9, 10],
             [9, 10, 11, 12]]
        );
        let res = Matrix::from(
            [[25, 28, 31, 34],
             [57, 64, 71, 78],
             [89, 100, 111, 122]]
        );

        assert_eq!(x * y, res);
    }

    #[test]
    fn matrix_vector_multiply() {
        let a = Matrix::from(
            [[1, -1, 2],
             [0, -3, 1]]
        );
        let x = Vector::from([2, 1, 0]);

        assert_eq!(a * x, Vector::from([1, -3]));
    }

    #[test]
    fn transpose() {
        let x = Matrix::from(
            [[1, 2, 3],
             [4, 5, 6]]
        );
        let y = Matrix::from(
            [[1, 4],
             [2, 5],
             [3, 6]]
        );

        assert_eq!(x.transpose(), y);
    }
}
