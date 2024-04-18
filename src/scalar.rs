use crate::numeric::Numeric;
use crate::tensor::{IndexError, Tensor};
use std::ops::MulAssign;

#[derive(Debug)]
pub struct Scalar<T: Numeric> {
    val: T,
}

impl<T: Numeric> From<T> for Scalar<T> {
    fn from(val: T) -> Self {
        Scalar { val }
    }
}

impl<T: Numeric> Tensor<T, 0> for Scalar<T> {
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

    fn next_idx(&self, _idx: [usize; 0]) -> Option<[usize; 0]> {
        None
    }
}

impl<T: Numeric> PartialEq for Scalar<T> {
    fn eq(&self, other: &Self) -> bool {
        self.get([]) == other.get([])
    }
}

impl<T: Numeric> Eq for Scalar<T> {}

impl<T: Numeric> MulAssign<T> for Scalar<T> {
    fn mul_assign(&mut self, other: T) {
        self.update(|n| n * other);
    }
}
