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
}

pub trait ShapedTensor<const R: usize, const S: TensorShape>: Tensor {
    fn rank() -> usize {
        R
    }
    fn shape() -> [usize; R] {
        let mut s = [0; R];
        s[..].copy_from_slice(&S[..R]);

        s
    }
    fn stride() -> [usize; R] {
        let mut res = [0; R];
        for (dim, item) in res.iter_mut().enumerate() {
            *item = S[(dim + 1)..R].iter().product();
        }

        res
    }
    fn get(&self, idx: [usize; R]) -> <Self as Tensor>::T;
    fn set(&self, idx: [usize; R], val: <Self as Tensor>::T);

    fn from_fn<F>(cb: F) -> Self
    where
        F: Fn([usize; R]) -> <Self as Tensor>::T;
}

pub struct TensorIterator<'a, const R: usize, const S: TensorShape, Tn>
where
    Tn: ShapedTensor<R, S>,
{
    t: &'a Tn,
    done: bool,
    cur: [usize; R],
}

impl<'a, const R: usize, const S: TensorShape, Tn> TensorIterator<'a, R, S, Tn>
where
    Tn: ShapedTensor<R, S>,
{
    pub fn new(t: &'a Tn) -> Self {
        Self {
            t,
            cur: [0; R],
            done: false,
        }
    }
}

impl<'a, const R: usize, const S: TensorShape, Tn> Iterator for TensorIterator<'a, R, S, Tn>
where
    Tn: ShapedTensor<R, S>,
{
    type Item = Tn::T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let item = self.t.get(self.cur);
        if R == 0 {
            self.done = true;
            return Some(item);
        }

        self.cur[R - 1] += 1;
        for dim in (0..R).rev() {
            if self.cur[dim] == S[dim] {
                if dim == 0 {
                    self.done = true;
                    break;
                }
                self.cur[dim] = 0;
                self.cur[dim - 1] += 1;
            }
        }

        Some(item)
    }
}
