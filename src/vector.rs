use std::ops::MulAssign;
use num::Num;
use crate::tensor::{Storage, Tensor};

#[derive(Debug)]
pub struct Vector<T: Num + Copy, const N: usize> {
    vals: Storage<T, N>,
}

impl<T: Num + Copy, const N: usize> From<[T; N]> for Vector<T, N> {
    fn from(vals: [T; N]) -> Self {
        Vector {
            vals: Storage::from(vals),
        }
    }
}

impl<T: Num + Copy, const N: usize> Tensor<T, 1> for Vector<T, N> {
    fn shape(&self) -> [usize; 1] {
        [N]
    }
}

impl<T: Num + Copy, const N: usize> MulAssign<T> for Vector<T, N> {
    fn mul_assign(&mut self, other: T) {
        self.vals.elem_mul(other);
    }
}
