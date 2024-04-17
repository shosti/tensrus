use std::ops::MulAssign;
use num::Num;
use crate::tensor::{Tensor, Storage};

#[derive(Debug)]
pub struct Scalar<T: Num + Copy> {
    val: Storage<T, 1>,
}

impl<T: Num + Copy> From<T> for Scalar<T> {
    fn from(val: T) -> Self {
        Scalar { val: Storage::from([val]) }
    }
}

impl<T: Num + Copy> Tensor<T, 0> for Scalar<T> {
    fn shape(&self) -> [usize; 0] {
        []
    }
}

impl<T: Num + Copy> MulAssign<T> for Scalar<T> {
    fn mul_assign(&mut self, other: T) {
        self.val.elem_mul(other);
    }
}
