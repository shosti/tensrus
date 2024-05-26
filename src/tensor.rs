use num::{One, Zero};
use std::any::Any;
use std::cmp::PartialEq;
use std::fmt::Debug;
use std::iter::Map;
use std::ops::{Add, Index, Mul};

use crate::numeric::Numeric;
use crate::scalar::Scalar;
use crate::slice::Slice;

#[derive(Debug, PartialEq)]
pub struct IndexError {}

pub type Shape = [usize; 5];

#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub enum Transpose {
    #[default]
    None,
    Transposed,
}

impl Transpose {
    pub fn transpose(self) -> Transpose {
        match self {
            Transpose::None => Transpose::Transposed,
            Transpose::Transposed => Transpose::None,
        }
    }

    pub fn is_transposed(self) -> bool {
        self == Transpose::Transposed
    }

    pub fn to_blas(&self) -> u8 {
        match self {
            Transpose::None => b'T',
            Transpose::Transposed => b'N',
        }
    }
}

pub const fn vector_shape(n: usize) -> Shape {
    [n; 5]
}

pub const fn transpose_shape(r: usize, s: Shape) -> Shape {
    let mut out = [0; 5];
    let mut i = 0;
    while i < r {
        out[i] = s[r - i - 1];
        i += 1;
    }

    out
}

pub const fn num_elems(r: usize, s: Shape) -> usize {
    let mut dim = 0;
    let mut n = 1;

    while dim < r {
        n *= s[dim];
        dim += 1;
    }

    n
}

pub const fn shape_dim(s: Shape, i: usize) -> usize {
    s[i]
}

pub const fn downrank(r: usize, s: Shape, n: usize) -> Shape {
    if n > r {
        panic!("downranking to negative rank");
    }
    let mut new_shape = [0; 5];
    let mut i = 0;
    while i < r - n {
        new_shape[i] = s[i + n];
        i += 1;
    }

    new_shape
}

pub const fn stride(r: usize, s: Shape) -> [usize; 5] {
    let mut res = [0; 5];
    let mut dim = 0;

    while dim < r {
        let mut i = dim + 1;
        let mut cur = 1;
        while i < r {
            cur *= s[i];
            i += 1;
        }
        res[dim] = cur;

        dim += 1;
    }

    res
}

pub trait BasicTensor<T: Numeric>: Debug + for<'a> Index<&'a [usize], Output = T> {
    fn as_any(&self) -> &dyn Any;
    fn as_any_boxed(self: Box<Self>) -> Box<dyn Any>;
    fn num_elems(&self) -> usize;
    fn clone_boxed(&self) -> Box<dyn BasicTensor<T>>;

    // Returns a new tensor of zeros with the same shape as self
    fn zeros_with_shape(&self) -> Box<dyn BasicTensor<T>>;

    // Returns a new tensor of ones with the same shape as self
    fn ones_with_shape(&self) -> Box<dyn BasicTensor<T>>;

    fn add(self: Box<Self>, other: &dyn BasicTensor<T>, scale: T) -> Box<dyn BasicTensor<T>>;
}

pub trait Tensor:
    BasicTensor<Self::T>
    + Clone
    + for<'a> Add<&'a Self, Output = Self>
    + Mul<Self::T, Output = Self>
    + for<'a> Index<&'a Self::Idx, Output = Self::T>
    + PartialEq
    + 'static
{
    type T: Numeric;
    type Idx: AsRef<[usize]> + Copy + 'static;

    fn repeat(n: Self::T) -> Self;
    fn zeros() -> Self {
        Self::repeat(Self::T::zero())
    }
    fn ones() -> Self {
        Self::repeat(Self::T::one())
    }
    fn from_fn(f: impl Fn(&Self::Idx) -> Self::T) -> Self {
        Self::zeros().map(|idx, _| f(idx))
    }
    fn num_elems() -> usize;

    fn iter(&self) -> TensorIterator<Self> {
        TensorIterator::new(self)
    }

    fn ref_from_basic(from: &dyn BasicTensor<Self::T>) -> &Self {
        let any_ref = from.as_any();
        any_ref.downcast_ref().unwrap()
    }

    fn from_basic(from: &dyn BasicTensor<Self::T>) -> Self {
        if from.num_elems() != <Self as Tensor>::num_elems() {
            panic!("cannot create a Tensor from a BasicTensor unless the number of elements is identical");
        }

        Self::from_fn(|idx| from[idx.as_ref()])
    }

    fn from_basic_boxed(from: Box<dyn BasicTensor<Self::T>>) -> Box<Self> {
        from.as_any_boxed().downcast().unwrap()
    }

    fn map(self, f: impl Fn(&Self::Idx, Self::T) -> Self::T) -> Self;
    fn set(self, idx: &Self::Idx, val: Self::T) -> Self;

    fn default_idx() -> Self::Idx;
    fn next_idx(&self, idx: &Self::Idx) -> Option<Self::Idx>;

    fn sum(&self) -> Scalar<Self::T> {
        let s: Self::T = self.iter().values().sum();
        Scalar::from(s)
    }

    // Operations
    fn relu(self) -> Self {
        self.map(|_, val| {
            if val < Self::T::zero() {
                Self::T::zero()
            } else {
                val
            }
        })
    }
}

pub trait SlicedTensor<T: Numeric, const R: usize, const S: Shape> {
    fn try_slice<const D: usize>(
        &self,
        idx: [usize; D],
    ) -> Result<Slice<T, { R - D }, { downrank(R, S, D) }>, IndexError>;

    fn slice<const D: usize>(&self, idx: [usize; D]) -> Slice<T, { R - D }, { downrank(R, S, D) }> {
        self.try_slice(idx).unwrap()
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

    pub fn values(self) -> Map<TensorIterator<'a, Tn>, impl FnMut((Tn::Idx, Tn::T)) -> Tn::T> {
        self.map(|(_, v)| v)
    }
}

impl<'a, Tn> Iterator for TensorIterator<'a, Tn>
where
    Tn: Tensor,
{
    type Item = (Tn::Idx, Tn::T);

    fn next(&mut self) -> Option<Self::Item> {
        match &self.cur {
            None => None,
            Some(idx) => {
                let cur_idx = *idx;
                let item = (cur_idx, self.t[&cur_idx]);
                self.cur = self.t.next_idx(&cur_idx);

                Some(item)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_downrank() {
        let s1 = [7, 6, 5, 0, 0];
        assert_eq!(downrank(3, s1, 0), s1);
        assert_eq!(downrank(3, s1, 1), [6, 5, 0, 0, 0]);
        assert_eq!(downrank(3, s1, 2), [5, 0, 0, 0, 0]);
        assert_eq!(downrank(3, s1, 3), [0, 0, 0, 0, 0]);
    }
}
