use crate::tensor::{IndexError, Tensor};
use num::Num;
use std::ops::MulAssign;

#[derive(Debug)]
pub struct Scalar<T: Num + Copy> {
    val: T,
}

impl<T: Num + Copy> From<T> for Scalar<T> {
    fn from(val: T) -> Self {
        Scalar { val }
    }
}

impl<T: Num + Copy> Tensor<T, 0> for Scalar<T> {
    type Transpose = Self;

    fn from_fn<F>(mut cb: F) -> Self
    where
        F: FnMut([usize; 0]) -> T,
    {
        Self { val: cb([]) }
    }

    fn shape(&self) -> [usize; 0] {
        []
    }

    fn get(&self, _idx: [usize; 0]) -> Result<T, IndexError> {
        Ok(self.val)
    }

    fn set(&mut self, _idx: [usize; 0], val: T) -> Result<(), IndexError> {
        self.val = val;
        Ok(())
    }

    fn transpose(&self) -> Self::Transpose {
        Self::Transpose { val: self.val }
    }
}

impl<T: Num + Copy> PartialEq for Scalar<T> {
    fn eq(&self, other: &Self) -> bool {
        self.get([]) == other.get([])
    }
}

impl<T: Num + Copy> Eq for Scalar<T> {}

impl<T: Num + Copy> MulAssign<T> for Scalar<T> {
    fn mul_assign(&mut self, other: T) {
        self.val = self.val * other;
    }
}
