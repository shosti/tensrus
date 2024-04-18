use crate::numeric::Numeric;
use std::ops::{FnMut, MulAssign};

pub trait Tensor<T: Numeric, const R: usize>: MulAssign<T> + Eq {
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
    fn next_idx(&self, idx: [usize; R]) -> Option<[usize; R]>;

    fn update<F>(&mut self, mut cb: F)
    where
        F: FnMut(T) -> T,
    {
        let mut iter = Some([0; R]);
        while let Some(idx) = iter {
            let cur = self.get(idx).unwrap();
            self.set(idx, cb(cur)).unwrap();
            iter = self.next_idx(idx);
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct IndexError {}
