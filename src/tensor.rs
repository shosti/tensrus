use crate::numeric::Numeric;
use crate::scalar::Scalar;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, IndexMut, Mul, MulAssign};

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

pub trait ShapedTensor<T: Numeric, const R: usize, const S: TensorShape>:
    IndexMut<[usize; R], Output = T>
{
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
}

pub trait TensorOps<T: Numeric>:
    Add<T>
    + Add<Scalar<T>>
    + AddAssign<T>
    + Mul<T>
    + Mul<Scalar<T>>
    + MulAssign<T>
    + Debug
    + Clone
    + 'static
{
    fn zeros() -> Self;
    fn update(&mut self, f: &dyn Fn(T) -> T);
    fn update_zip(&mut self, other: &Self, f: &dyn Fn(T, T) -> T);
}

pub struct TensorIterator<'a, T: Numeric, const R: usize, const S: TensorShape> {
    t: &'a dyn ShapedTensor<T, R, S>,
    done: bool,
    cur: [usize; R],
}

impl<'a, T: Numeric, const R: usize, const S: TensorShape> TensorIterator<'a, T, R, S> {
    pub fn new(t: &'a dyn ShapedTensor<T, R, S>) -> Self {
        Self {
            t,
            cur: [0; R],
            done: false,
        }
    }
}

impl<'a, T: Numeric, const R: usize, const S: TensorShape> Iterator
    for TensorIterator<'a, T, R, S>
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let item = self.t[self.cur];
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
