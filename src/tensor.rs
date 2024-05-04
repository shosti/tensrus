use std::any::Any;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, MulAssign};

use crate::numeric::Numeric;
use crate::scalar::Scalar;

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

pub trait BasicTensor: Debug {
    fn as_any(&self) -> &dyn Any;
}

pub trait Tensor:
    BasicTensor
    + Add
    + AddAssign
    + Mul<Self::T>
    + Mul<Scalar<Self::T>>
    + MulAssign<Self::T>
    + Clone
    + 'static
{
    type T: Numeric;
    type Idx: Copy + 'static;

    fn zeros() -> Self;
    fn deep_clone(&self) -> Self;
    fn update<F: Fn(Self::T) -> Self::T>(&mut self, f: F);
    fn update_zip<F: Fn(Self::T, Self::T) -> Self::T>(&mut self, other: &Self, f: F);
    fn update_zip2<F: Fn(Self::T, Self::T, Self::T) -> Self::T>(
        &mut self,
        a: &Self,
        b: &Self,
        f: F,
    );

    fn default_idx() -> Self::Idx;
    fn next_idx(idx: Self::Idx) -> Option<Self::Idx>;

    fn get(&self, idx: Self::Idx) -> Self::T;

    fn set(&self, idx: Self::Idx, val: Self::T);

    fn from_fn<F>(cb: F) -> Self
    where
        F: Fn(Self::Idx) -> Self::T;
}

pub struct TensorIterator<'a, Tn>
where
    Tn: Tensor,
{
    t: &'a Tn,
    cur: Option<Tn::Idx>,
}

impl<'a, Tn> TensorIterator<'a, Tn>
where
    Tn: Tensor,
{
    pub fn new(t: &'a Tn) -> Self {
        Self {
            t,
            cur: Some(Tn::default_idx()),
        }
    }
}

impl<'a, Tn> Iterator for TensorIterator<'a, Tn>
where
    Tn: Tensor,
{
    type Item = Tn::T;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.cur {
            None => None,
            Some(idx) => {
                let cur_idx = *idx;
                let item = self.t.get(cur_idx);
                self.cur = Tn::next_idx(cur_idx);

                Some(item)
            }
        }
    }
}
