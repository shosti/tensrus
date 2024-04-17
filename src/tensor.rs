use num::Num;
use std::cell::RefCell;
use std::ops::{FnMut, Index, MulAssign};
use std::rc::Rc;

pub trait Tensor<T: Num + Copy, const R: usize>: MulAssign<T> {
    fn rank(&self) -> usize {
        R
    }

    fn shape(&self) -> [usize; R];
}

#[derive(Debug)]
pub struct Storage<T: Num + Copy, const N: usize> {
    vals: Rc<RefCell<[T; N]>>,
}

impl<T: Num + Copy, const N: usize> Storage<T, N> {
    pub fn from_fn<F>(cb: F) -> Self
    where
        F: FnMut(usize) -> T,
    {
        Self {
            vals: Rc::new(RefCell::new(std::array::from_fn(cb))),
        }
    }

    pub fn elem_mul(&mut self, other: T) {
        let mut vals = self.vals.borrow_mut();

        for i in 0..N {
            vals[i] = vals[i] * other;
        }
    }
}

impl<T: Num + Copy, const N: usize> From<[T; N]> for Storage<T, N> {
    fn from(vals: [T; N]) -> Self {
        Storage {
            vals: Rc::new(RefCell::new(vals)),
        }
    }
}
