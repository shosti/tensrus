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

// Marker trait for the purposes of the derive macro
pub trait Tensor {}

pub trait ShapedTensor<T: Numeric, const R: usize, const S: TensorShape> {
    fn rank(&self) -> usize {
        R
    }
    fn shape(&self) -> [usize; R] {
        let mut s = [0; R];
        for i in 0..R {
            s[i] = S[i];
        }

        s
    }
    fn get(&self, idx: [usize; R]) -> T;
    fn set(&self, idx: [usize; R], val: T);
}

pub trait TensorOps<T: Numeric>:
    Add + AddAssign + Mul<T> + Mul<Scalar<T>> + MulAssign<T> + Debug + Clone + 'static
{
    fn zeros() -> Self;
    fn update<F: Fn(T) -> T>(&mut self, f: F);
    fn update_zip<F: Fn(T, T) -> T>(&mut self, other: &Self, f: F);
}

pub struct TensorIterator<'a, T: Numeric, const R: usize, const S: TensorShape, Tn>
where
    Tn: ShapedTensor<T, R, S>,
{
    _ignored: std::marker::PhantomData<T>,
    t: &'a Tn,
    done: bool,
    cur: [usize; R],
}

impl<'a, T: Numeric, const R: usize, const S: TensorShape, Tn> TensorIterator<'a, T, R, S, Tn>
where
    Tn: ShapedTensor<T, R, S>,
{
    pub fn new(t: &'a Tn) -> Self {
        Self {
            _ignored: std::marker::PhantomData,
            t,
            cur: [0; R],
            done: false,
        }
    }
}

impl<'a, T: Numeric, const R: usize, const S: TensorShape, Tn> Iterator
    for TensorIterator<'a, T, R, S, Tn>
where
    Tn: ShapedTensor<T, R, S>,
{
    type Item = T;

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
