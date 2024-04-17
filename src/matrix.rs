use crate::tensor::{Storage, Tensor};
use num::Num;
use std::ops::MulAssign;

#[derive(Debug)]
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

impl<T: Num + Copy, const N: usize, const M: usize> Tensor<T, 2> for Matrix<T, N, M>
where
    [(); N * M]:,
{
    fn shape(&self) -> [usize; 2] {
        [N, M]
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
