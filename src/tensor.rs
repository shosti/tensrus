use num::{One, Zero};
use std::fmt::Debug;
use std::iter::Map;
use std::ops::{Add, Index, Mul};

use crate::numeric::Numeric;
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

pub trait BasicTensor<T: Numeric>: Debug + for<'a> Index<&'a [usize], Output = T> {}

pub trait Tensor:
    BasicTensor<Self::T>
    + Clone
    + for<'a> Add<&'a Self, Output = Self>
    + Mul<Self::T, Output = Self>
    + for<'a> Index<&'a Self::Idx, Output = Self::T>
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
    fn from_fn(f: impl Fn(&Self::Idx) -> Self::T) -> Self {
        Self::zeros().map(|idx, _| f(idx))
    }
    fn iter(&self) -> TensorIterator<Self> {
        TensorIterator::new(self)
    }

    fn map(self, f: impl Fn(&Self::Idx, Self::T) -> Self::T) -> Self;
    fn set(self, idx: &Self::Idx, val: Self::T) -> Self;

    fn default_idx() -> Self::Idx;
    fn next_idx(&self, idx: &Self::Idx) -> Option<Self::Idx>;

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
