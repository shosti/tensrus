use num::Num;
use std::ops::{FnMut, MulAssign};

pub trait Tensor<T: Num + Copy, const R: usize>: MulAssign<T> + Eq {
    type Transpose;

    fn from_fn<F>(cb: F) -> Self
    where
        F: FnMut([usize; R]) -> T;
    fn zeros() -> Self
    where
        Self: Sized,
    {
        Self::from_fn(|_| T::zero())
    }

    fn rank(&self) -> usize {
        R
    }
    fn shape(&self) -> [usize; R];
    fn get(&self, idx: [usize; R]) -> Result<T, IndexError>;
    fn set(&mut self, idx: [usize; R], val: T) -> Result<(), IndexError>;
    fn transpose(&self) -> Self::Transpose;
}

#[derive(Debug, PartialEq)]
pub struct IndexError {}
