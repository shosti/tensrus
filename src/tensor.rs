use crate::numeric::Numeric;
use crate::scalar::Scalar;
use std::ops::{Add, AddAssign, Mul, MulAssign};

#[derive(Debug, PartialEq)]
pub struct IndexError {}

#[derive(Debug, PartialEq)]
pub struct ShapeError {}

pub type TensorShape = [usize; 5];

pub const fn vector_shape(n: usize) -> TensorShape {
    [n; 5]
}

pub const fn num_elems(r: usize, s: TensorShape) -> usize {
    let mut dim = 0;
    let mut n = 1;

    while dim < r {
        n *= s[dim];
        dim += 1;
    }

    n
}

pub const fn shape_dim(s: TensorShape, i: usize) -> usize {
    s[i]
}

pub trait Tensor<T: Numeric, const R: usize> {
    fn from_fn<F>(cb: F) -> Self
    where
        F: FnMut([usize; R]) -> T,
        Self: Sized;

    fn rank(&self) -> usize {
        R
    }
    fn shape(&self) -> [usize; R];
    fn get(&self, idx: &[usize; R]) -> Result<T, IndexError>;
    fn get_at_idx(&self, i: usize) -> Result<T, IndexError>;
    fn update(&self, f: &dyn Fn(T) -> T);
}

pub trait TensorOps<T: Numeric>:
    Add<T> + Add<Scalar<T>> + AddAssign<T> + Mul<T> + Mul<Scalar<T>> + MulAssign<T>
{
}

pub struct TensorIterator<'a, T: Numeric, const R: usize> {
    t: &'a dyn Tensor<T, R>,
    cur: usize,
}

impl<'a, T: Numeric, const R: usize> TensorIterator<'a, T, R> {
    pub fn new(t: &'a dyn Tensor<T, R>) -> Self {
        Self { t, cur: 0 }
    }
}

impl<'a, T: Numeric, const R: usize> Iterator
    for TensorIterator<'a, T, R>
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.t.get_at_idx(self.cur) {
            Ok(val) => {
                self.cur += 1;
                Some(val)
            }
            Err(_) => None,
        }
    }
}
