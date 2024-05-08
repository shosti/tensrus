use num::{One, Zero};
use std::fmt::Debug;
use std::ops::{Add, Index, Mul};

use crate::numeric::Numeric;
use crate::shape::Shape;
use crate::slice::Slice;

#[derive(Debug, PartialEq)]
pub struct IndexError {}

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

pub trait Tensor: Clone + 'static
// Debug + Clone + for<'a> Add<&'a Self, Output = Self> + Mul<Self::T> + 'static
{
    type T: Numeric;
    const S: Shape;

    fn repeat(n: Self::T) -> Self;
    fn zeros() -> Self {
        Self::repeat(Self::T::zero())
    }
    fn ones() -> Self {
        Self::repeat(Self::T::one())
    }
    fn from_fn(f: impl Fn([usize; Self::S.rank()]) -> Self::T) -> Self
    where
        [(); Self::S.rank()]:;


    fn map(self, f: impl Fn(Self::T) -> Self::T) -> Self;
    fn zip<'a>(self, other: &'a Self) -> TensorZipper<'a, Self> {
        TensorZipper::new(self, other)
    }
    fn reduce<'a>(self, others: Vec<&'a Self>, f: impl Fn(Vec<Self::T>) -> Self::T) -> Self;
    fn try_slice<'a, const D: usize>(
        &'a self,
        idx: [usize; D],
    ) -> Result<Slice<'a, Self::T, { Self::S.downrank(D) }>, IndexError>;

    fn slice<'a, const D: usize>(
        &'a self,
        idx: [usize; D],
    ) -> Slice<'a, Self::T, { Self::S.downrank(D) }> {
        self.try_slice(idx).unwrap()
    }

    fn nth_elem(&self, i: usize) -> Result<Self::T, IndexError>;
}

pub struct TensorZipper<'a, Tn: Tensor> {
    t: Tn,
    others: Vec<&'a Tn>,
}

impl<'a, Tn: Tensor> TensorZipper<'a, Tn> {
    pub fn new(t: Tn, other: &'a Tn) -> Self {
        Self {
            t,
            others: vec![other],
        }
    }

    pub fn zip(self, other: &'a Tn) -> Self {
        let mut others = self.others;
        others.push(other);
        Self { t: self.t, others }
    }

    pub fn map(self, f: impl Fn(Vec<Tn::T>) -> Tn::T) -> Tn {
        self.t.reduce(self.others, f)
    }
}

pub struct TensorIterator<'a, Tn>
where
    Tn: Tensor,
{
    t: &'a Tn,
    cur: usize,
}

impl<'a, Tn> TensorIterator<'a, Tn>
where
    Tn: Tensor,
{
    pub fn new(t: &'a Tn) -> Self {
        Self {
            t,
            cur: 0,
        }
    }
}

impl<'a, Tn> Iterator for TensorIterator<'a, Tn>
where
    Tn: Tensor,
{
    type Item = Tn::T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.t.nth_elem(self.cur) {
            Ok(val) => {
                self.cur += 1;
                Some(val)
            },
            Err(IndexError {}) => {
                None
            }
        }
    }
}
