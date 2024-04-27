use crate::numeric::Numeric;
use crate::scalar::Scalar;
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

pub trait Tensor<T: Numeric, const R: usize, const S: TensorShape>:
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
    fn get(&self, idx: &[usize; R]) -> Result<T, IndexError>;
    fn set(&mut self, idx: &[usize; R], val: T) -> Result<(), IndexError>;
    fn update(&mut self, f: &dyn Fn(T) -> T);
}

pub trait TensorOps<T: Numeric>:
    Add<T> + Add<Scalar<T>> + AddAssign<T> + Mul<T> + Mul<Scalar<T>> + MulAssign<T>
{
}

pub struct TensorIterator<'a, T: Numeric, const R: usize, const S: TensorShape> {
    t: &'a dyn Tensor<T, R, S>,
    done: bool,
    cur: [usize; R],
}

impl<'a, T: Numeric, const R: usize, const S: TensorShape> TensorIterator<'a, T, R, S> {
    pub fn new(t: &'a dyn Tensor<T, R, S>) -> Self {
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

        let item = self.t.get(&self.cur).unwrap();
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
