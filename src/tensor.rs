use std::ops::MulAssign;
use num::Num;

pub trait Tensor<T: Num + Copy, const R: usize>: MulAssign<T> {
    fn rank(&self) -> usize {
        R
    }

    fn shape(&self) -> [usize; R];
}

#[derive(Debug)]
pub struct Scalar<T: Num + Copy> {
    val: T,
}

impl<T: Num + Copy> From<T> for Scalar<T> {
    fn from(val: T) -> Self {
        Scalar { val: val }
    }
}

impl<T: Num + Copy> Tensor<T, 0> for Scalar<T> {
    fn shape(&self) -> [usize; 0] {
        []
    }
}

impl<T: Num + Copy> MulAssign<T> for Scalar<T> {
    fn mul_assign(&mut self, other: T) {
        self.val = self.val * other;
    }
}

#[derive(Debug)]
pub struct Vector<T: Num + Copy, const N: usize> {
    shape: (usize,),
    val: [T; N],
}

impl<T: Num + Copy, const N: usize> From<[T; N]> for Vector<T, N> {
    fn from(vals: [T; N]) -> Self {
        Vector {
            shape: (N,),
            val: vals,
        }
    }
}

impl<T: Num + Copy, const N: usize> Tensor<T, 1> for Vector<T, N> {
    fn shape(&self) -> [usize; 1] {
        [self.shape.0]
    }
}

impl<T: Num + Copy, const N: usize> MulAssign<T> for Vector<T, N> {
    fn mul_assign(&mut self, other: T) {
        for i in 0..N {
            self.val[i] = self.val[i] * other;
        }
    }
}

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
