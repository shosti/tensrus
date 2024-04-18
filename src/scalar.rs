use crate::tensor::{IndexError, Storage, Tensor};
use num::Num;
use std::ops::MulAssign;

#[derive(Debug)]
pub struct Scalar<T: Num + Copy> {
    val: Storage<T, 1>,
}

impl<T: Num + Copy> From<T> for Scalar<T> {
    fn from(val: T) -> Self {
        Scalar {
            val: Storage::from([val]),
        }
    }
}

impl<T: Num + Copy> Tensor<T, 0> for Scalar<T> {
    fn shape(&self) -> [usize; 0] {
        []
    }

    fn get(&self, _idx: [usize; 0]) -> Result<T, IndexError> {
        Ok(self.val.get(0))
    }

    fn set(&mut self, _idx: [usize; 0], val: T) -> Result<(), IndexError> {
        self.val.set(0, val);
        Ok(())
    }
}

impl<T: Num + Copy> MulAssign<T> for Scalar<T> {
    fn mul_assign(&mut self, other: T) {
        self.val.elem_mul(other);
    }
}
