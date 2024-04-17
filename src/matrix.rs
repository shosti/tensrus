use std::ops::MulAssign;
use num::Num;
use crate::tensor::Tensor;

#[derive(Debug)]
pub struct Matrix<T: Num + Copy, const N: usize, const M: usize>
where
    [(); N * M]:,
{
    shape: (usize, usize),
    val: [T; N * M],
}

impl<T: Num + Copy, const N: usize, const M: usize> From<[[T; N]; M]> for Matrix<T, N, M>
where
    [(); N * M]:,
{
    fn from(vals: [[T; N]; M]) -> Self {
        let mut ret: Matrix<T, N, M> = Matrix {
            shape: (N, M),
            val: std::array::from_fn(|_| T::zero()),
        };
        for i in 0..M {
            for j in 0..N {
                ret.val[(i * N) + j] = vals[i][j];
            }
        }

        ret
    }
}

impl<T: Num + Copy, const N: usize, const M: usize> Tensor<T, 2> for Matrix<T, N, M> where [(); N * M]: {
    fn shape(&self) -> [usize; 2] {
        [self.shape.0, self.shape.1]
    }
}

impl<T: Num + Copy, const N: usize, const M: usize> MulAssign<T> for Matrix<T, N, M> where [(); N * M]: {
    fn mul_assign(&mut self, other: T) {
        for i in 0..N {
            self.val[i] = self.val[i] * other;
        }
    }
}
