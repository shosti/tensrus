use num::Num;
use std::cell::RefCell;
use std::ops::{FnMut, MulAssign};
use std::rc::Rc;

pub trait Tensor<T: Num + Copy, const R: usize>: MulAssign<T> + Eq {
    fn from_fn<F>(cb: F) -> Self
    where
        F: FnMut(usize) -> T;
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
}

#[derive(Debug)]
pub struct Storage<T: Num + Copy, const N: usize> {
    vals: Rc<RefCell<[T; N]>>,
}

#[derive(Debug, PartialEq)]
pub struct IndexError {}

impl<T: Num + Copy, const N: usize> Storage<T, N> {
    pub fn from_fn<F>(cb: F) -> Self
    where
        F: FnMut(usize) -> T,
    {
        Self {
            vals: Rc::new(RefCell::new(std::array::from_fn(cb))),
        }
    }

    pub fn zeros() -> Self {
        Self {
            vals: Rc::new(RefCell::new(std::array::from_fn(|_| T::zero()))),
        }
    }

    pub fn get(&self, idx: usize) -> T {
        self.vals.borrow()[idx]
    }

    pub fn elem_mul(&mut self, other: T) {
        let mut vals = self.vals.borrow_mut();

        for i in 0..N {
            vals[i] = vals[i] * other;
        }
    }

    pub fn set(&mut self, idx: usize, val: T) {
        let mut vals = self.vals.borrow_mut();

        vals[idx] = val;
    }
}

impl<T: Num + Copy, const N: usize> From<[T; N]> for Storage<T, N> {
    fn from(vals: [T; N]) -> Self {
        Storage {
            vals: Rc::new(RefCell::new(vals)),
        }
    }
}
