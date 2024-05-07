use num::{One, Zero};
use std::fmt::Debug;
use std::ops::{Add, Mul};

use crate::numeric::Numeric;

#[derive(Debug, PartialEq)]
pub struct IndexError {}
// pub enum TensorShape2 {
//     Rank0([usize; 0),
//     Rank1([usize; 1]),
//     Rank2([usize; 2]),
//           Rank3([
// }


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

pub trait Tensor:
    Debug + Clone + for<'a> Add<&'a Self, Output = Self> + Mul<Self::T> + 'static
{
    type T: Numeric;
    type Idx: Copy + 'static;

    fn repeat(n: Self::T) -> Self;
    fn zeros() -> Self {
        Self::repeat(Self::T::zero())
    }
    fn ones() -> Self {
        Self::repeat(Self::T::one())
    }
    fn from_fn(f: impl Fn(Self::Idx) -> Self::T) -> Self;

    fn get(&self, idx: Self::Idx) -> Self::T;

    fn map(self, f: impl Fn(Self::T) -> Self::T) -> Self;
    fn zip<'a>(self, other: &'a Self) -> TensorZipper<'a, Self> {
        TensorZipper::new(self, other)
    }
    fn set(self, idx: Self::Idx, val: Self::T) -> Self;
    fn reduce<'a>(self, others: Vec<&'a Self>, f: impl Fn(Vec<Self::T>) -> Self::T) -> Self;

    fn default_idx() -> Self::Idx;
    fn next_idx(idx: Self::Idx) -> Option<Self::Idx>;
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
