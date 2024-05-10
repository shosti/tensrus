use num::{One, Zero};
use std::fmt::Debug;
use std::iter::Map;
use std::ops::{Add, Index, Mul};

use crate::numeric::Numeric;
use crate::slice::Slice;

#[derive(Debug, PartialEq)]
pub struct IndexError {}

pub type Shape = [usize; 5];

pub const fn vector_shape(n: usize) -> Shape {
    [n; 5]
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

pub fn stride<const R: usize, const S: Shape>() -> [usize; R] {
    let mut res = [0; R];
    for (dim, item) in res.iter_mut().enumerate() {
        *item = S[(dim + 1)..R].iter().product();
    }

    res
}

pub trait Tensor:
    Debug
    + Clone
    + for<'a> Add<&'a Self, Output = Self>
    + Mul<Self::T, Output = Self>
    + Index<Self::Idx, Output = Self::T>
    + 'static
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
    fn from_fn(f: impl Fn(Self::Idx) -> Self::T) -> Self {
        Self::zeros().map(|_, idx| f(idx))
    }

    fn map(self, f: impl Fn(Self::T, Self::Idx) -> Self::T) -> Self;
    fn set(self, idx: Self::Idx, val: Self::T) -> Self;

    fn default_idx() -> Self::Idx;
    fn next_idx(&self, idx: Self::Idx) -> Option<Self::Idx>;
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

    pub fn values(self) -> Map<TensorIterator<'a, Tn>, &'a dyn Fn((Tn::T, Tn::Idx)) -> Tn::T> {
        self.map(&|(v, _)| v)
    }
}

impl<'a, Tn> Iterator for TensorIterator<'a, Tn>
where
    Tn: Tensor,
{
    type Item = (Tn::T, Tn::Idx);

    fn next(&mut self) -> Option<Self::Item> {
        match &self.cur {
            None => None,
            Some(idx) => {
                let cur_idx = *idx;
                let item = (self.t[cur_idx], cur_idx);
                self.cur = self.t.next_idx(cur_idx);

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
