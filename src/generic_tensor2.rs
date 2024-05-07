use num::ToPrimitive;

use crate::numeric::Numeric;
use crate::scalar::Scalar;
use crate::shape::Shape;
use crate::tensor::{IndexError, Tensor};
use std::ops::{Add, Mul};

#[derive(Debug, Clone)]
pub struct GenericTensor<T: Numeric, const S: Shape> {
    storage: Vec<T>,
}

impl<T: Numeric, const S: Shape> GenericTensor<T, S> {
    fn storage_size() -> usize {
        S.len()
    }

    fn idx_from_storage_idx(idx: usize) -> Result<[usize; S.rank()], IndexError> {
        if idx >= Self::storage_size() {
            return Err(IndexError {});
        }

        let mut res = [0; S.rank()];
        let mut i = idx;
        let stride: [usize; S.rank()] = S.stride();

        for (dim, item) in res.iter_mut().enumerate() {
            let s: usize = stride[dim];
            let cur = i / s;
            *item = cur;
            i -= cur * s;
        }
        debug_assert!(i == 0);
        debug_assert!(Self::storage_idx(res).unwrap() == idx);

        Ok(res)
    }

    fn storage_idx(idx: [usize; S.rank()]) -> Result<usize, IndexError> {
        if S.rank() == 0 {
            return Ok(0);
        }

        let stride: [usize; S.rank()] = S.stride();
        let mut i = 0;
        for (dim, &cur) in idx.iter().enumerate() {
            if cur >= S[dim] {
                return Err(IndexError {});
            }
            i += stride[dim] * idx[dim];
        }

        Ok(i)
    }

    // pub fn reshape<const R2: usize, const S2: TensorShape>(self) -> GenericTensor<T, R2, S2>
    // where
    //     Assert<{ num_elems(R, S) == num_elems(R2, S2) }>: IsTrue,
    // {
    //     GenericTensor {
    //         storage: self.storage,
    //     }
    // }

    // pub fn subtensor(
    //     &self,
    //     i: usize,
    // ) -> Result<GenericTensor<T, { R - 1 }, { subtensor_shape(R, S) }>, IndexError> {
    //     if i >= S[0] {
    //         return Err(IndexError {});
    //     }

    //     let out: GenericTensor<T, { R - 1 }, { subtensor_shape(R, S) }> =
    //         GenericTensor::from_fn(|idx| {
    //             let mut self_idx = [i; R];
    //             self_idx[1..R].copy_from_slice(&idx[..(R - 1)]);
    //             self.storage[Self::storage_idx(self_idx).unwrap()]
    //         });
    //     Ok(out)
    // }
}

impl<T: Numeric, const S: Shape> Tensor for GenericTensor<T, S>
where
    [(); S.rank()]:,
{
    type T = T;
    type Idx = [usize; S.rank()];

    fn get(&self, idx: Self::Idx) -> T {
        match Self::storage_idx(idx) {
            Ok(i) => self.storage[i],
            Err(_e) => panic!("get: out of bounds"),
        }
    }

    fn set(self, idx: Self::Idx, val: T) -> Self {
        match Self::storage_idx(idx) {
            Ok(i) => {
                let mut storage = self.storage;
                storage[i] = val;
                Self { storage }
            }
            Err(_e) => panic!("set: out of bounds"),
        }
    }

    fn map(self, f: impl Fn(T) -> T) -> Self {
        let mut storage = self.storage;
        storage.iter_mut().for_each(|v| *v = f(*v));

        Self { storage }
    }

    fn reduce<'a>(self, others: Vec<&'a Self>, f: impl Fn(Vec<T>) -> T) -> Self {
        let mut storage = self.storage;
        storage.iter_mut().enumerate().for_each(|(i, v)| {
            let mut vals = vec![*v];
            for other in others.iter() {
                vals.push(other.storage[i]);
            }
            let out = f(vals);

            *v = out
        });

        Self { storage }
    }

    fn default_idx() -> Self::Idx {
        [0; S.rank()]
    }
    fn next_idx(idx: Self::Idx) -> Option<Self::Idx> {
        let mut cur = idx;
        cur[S.rank() - 1] += 1;
        for dim in (0..S.rank()).rev() {
            if cur[dim] == S[dim] {
                if dim == 0 {
                    return None;
                }
                cur[dim] = 0;
                cur[dim - 1] += 1;
            }
        }

        Some(cur)
    }

    fn repeat(n: T) -> Self {
        let storage = vec![n; Self::storage_size()];
        Self { storage }
    }

    fn from_fn(f: impl Fn(Self::Idx) -> T) -> Self {
        (0..Self::storage_size())
            .map(|i| f(Self::idx_from_storage_idx(i).unwrap()))
            .collect()
    }
}

impl<T: Numeric, const S: Shape, U: ToPrimitive> FromIterator<U> for GenericTensor<T, S>
where
    [(); S.rank()]:,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = U>,
    {
        let vals: Vec<T> = iter
            .into_iter()
            .map(|v| T::from(v).unwrap())
            .chain(std::iter::repeat(T::zero()))
            .take(Self::storage_size())
            .collect();
        Self { storage: vals }
    }
}

impl<T: Numeric, const S: Shape> Mul<T> for GenericTensor<T, S>
where
    [(); S.rank()]:,
{
    type Output = Self;

    fn mul(self, other: T) -> Self::Output {
        Self::from_fn(|idx| self.get(idx) * other)
    }
}

impl<T: Numeric, const S: Shape> Mul<Scalar<T>> for GenericTensor<T, S>
where
    [(); S.rank()]:,
{
    type Output = Self;

    fn mul(self, other: Scalar<T>) -> Self::Output {
        Self::from_fn(|idx| self.get(idx) * other.val())
    }
}

impl<'a, T: Numeric, const S: Shape> Add<&'a Self> for GenericTensor<T, S>
where
    [(); S.rank()]:,
{
    type Output = Self;

    fn add(self, other: &Self) -> Self::Output {
        self.zip(other).map(|vs| vs[0] + vs[1])
    }
}
