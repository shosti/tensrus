use crate::tensor::{IndexError, Storage, Tensor};
use num::Num;
use std::fmt::Display;
use std::ops::MulAssign;

pub struct Matrix<T: Num + Copy, const N: usize, const M: usize>
where
    [(); N * M]:,
{
    vals: Storage<T, { N * M }>,
}

impl<T: Num + Copy, const N: usize, const M: usize> From<[[T; N]; M]> for Matrix<T, N, M>
where
    [(); N * M]:,
{
    fn from(vals: [[T; N]; M]) -> Self {
        Self {
            vals: Storage::from_fn(|i| {
                let row = i / N;
                let col = i % N;

                vals[row][col]
            }),
        }
    }
}

impl<T: Num + Copy, const N: usize, const M: usize> Matrix<T, N, M>
where
    [(); N * M]:,
{
    pub fn get(&self, i: usize, j: usize) -> Result<T, IndexError> {
        if i >= M || j >= N {
            return Err(IndexError {});
        }

        Ok(self.vals.get((i * N) + j))
    }
}

impl<T: Num + Copy, const N: usize, const M: usize> Tensor<T, 2> for Matrix<T, N, M>
where
    [(); N * M]:,
{
    fn shape(&self) -> [usize; 2] {
        [M, N]
    }
}

impl<T: Num + Copy, const N: usize, const M: usize> MulAssign<T> for Matrix<T, N, M>
where
    [(); N * M]:,
{
    fn mul_assign(&mut self, other: T) {
        self.vals.elem_mul(other);
    }
}

impl<T: Num + Copy, const N: usize, const M: usize> PartialEq for Matrix<T, N, M>
where
    [(); N * M]:,
{
    fn eq(&self, other: &Self) -> bool {
        for i in 0..M {
            for j in 0..N {
                if self.get(i, j).unwrap() != other.get(i, j).unwrap() {
                    return false;
                }
            }
        }

        true
    }
}

impl<T: Num + Copy, const N: usize, const M: usize> Eq for Matrix<T, N, M> where [(); N * M]: {}

impl<T: Num + Copy + Display, const N: usize, const M: usize> std::fmt::Debug for Matrix<T, N, M>
where
    [(); N * M]:,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut repr = String::from(format!("{}x{} Matrix", N, M));
        repr.push_str(" {\n [");
        for i in 0..M {
            for j in 0..N {
                repr.push_str(&format!("{} ", self.get(i, j).unwrap()));
            }
            repr.push_str("\n");
        }
        repr.push_str("]\n");

        write!(f, "{}", repr)
    }
}

#[cfg(test)]
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
        assert_eq!(x.get(2, 1), Ok(5));
        assert_eq!(x.get(3, 2), Ok(3));
        assert_eq!(x.get(4, 1), Err(IndexError {}));
    }

    #[test]
    fn equality() {
        let x = Matrix::from(
            [[1, 2, 3],
             [4, 5, 6]]
        );
        let y = Matrix::from(
            [[1, 2, 3], [4, 5, 6]]
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
}
