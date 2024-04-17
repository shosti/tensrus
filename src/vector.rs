use std::ops::MulAssign;
use num::Num;
use crate::tensor::Tensor;

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
