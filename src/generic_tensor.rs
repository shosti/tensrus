use num::ToPrimitive;

use crate::numeric::Numeric;
// use crate::scalar::Scalar;
use crate::shape::Shape;
use crate::slice::Slice;
use crate::tensor::{IndexError, Tensor, TensorIterator};
use crate::type_assert::{Assert, IsTrue};
use std::ops::{Add, Index, Mul};

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct GenericTensor<T: Numeric, const S: Shape> {
    pub(crate) storage: Vec<T>,
}

impl<T: Numeric, const S: Shape> GenericTensor<T, S> {
    fn storage_size() -> usize {
        S.num_elems()
    }

    // Extra type param is to hack around the annoying S vs Self::S difference
    // :(
    fn idx_from_storage_idx<const S2: Shape>(idx: usize) -> Result<[usize; S2.rank()], IndexError> {
        if idx >= Self::storage_size() {
            return Err(IndexError {});
        }

        let mut res = [0; S2.rank()];
        let mut i = idx;
        let stride = S2.stride();

        for (dim, item) in res.iter_mut().enumerate() {
            let s: usize = stride[dim];
            let cur = i / s;
            *item = cur;
            i -= cur * s;
        }

        Ok(res)
    }

    pub fn reshape<const S2: Shape>(self) -> GenericTensor<T, S2>
    where
        Assert<{ S.num_elems() == S2.num_elems() }>: IsTrue,
    {
        GenericTensor {
            storage: self.storage,
        }
    }
}

impl<T: Numeric, const S: Shape> Tensor for GenericTensor<T, S> {
    type T = T;
    const S: Shape = S;

    fn try_slice<'a, const D: usize>(
        &'a self,
        idx: [usize; D],
    ) -> Result<Slice<'a, T, { Self::S.downrank(D) }>, IndexError> {
        Slice::new::<D, S>(&self.storage, idx)
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

    fn repeat(n: T) -> Self {
        let storage = vec![n; Self::storage_size()];
        Self { storage }
    }

    fn nth_elem(&self, i: usize) -> Result<Self::T, IndexError> {
        if i >= Self::storage_size() {
            Err(IndexError {})
        } else {
            Ok(self.storage[i])
        }
    }

    fn from_fn(f: impl Fn([usize; Self::S.rank()]) -> T) -> Self
    where
        [(); Self::S.rank()]:,
    {
        (0..Self::storage_size())
            .map(|i| {
                let idx: [usize; Self::S.rank()] = Self::idx_from_storage_idx(i).unwrap();
                f(idx)
            })
            .collect()
    }
}

impl<T: Numeric, const S: Shape, U: ToPrimitive> FromIterator<U> for GenericTensor<T, S> {
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

impl<'a, T: Numeric, const S: Shape> IntoIterator for &'a GenericTensor<T, S>
where
    [(); S.rank()]:,
{
    type Item = T;
    type IntoIter = TensorIterator<'a, GenericTensor<T, S>>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter::new(self)
    }
}

impl<T: Numeric, const S: Shape> Mul<T> for GenericTensor<T, S> {
    type Output = Self;

    fn mul(self, other: T) -> Self::Output {
        let mut storage = self.storage;
        storage.iter_mut().for_each(|x| *x *= other);
        Self { storage }
    }
}

// impl<T: Numeric, const S: Shape> Mul<&Scalar<T>> for GenericTensor<T, S>
// {
//     type Output = Self;

//     fn mul(self, other: &Scalar<T>) -> Self::Output {
//         self * other.val()
//     }
// }

impl<'a, T: Numeric, const S: Shape> Add<&'a Self> for GenericTensor<T, S> {
    type Output = Self;

    fn add(self, other: &Self) -> Self::Output {
        self.zip(other).map(|vs| vs[0] + vs[1])
    }
}

impl<T: Numeric, const S: Shape> Index<[usize; S.rank()]> for GenericTensor<T, S> {
    type Output = T;

    fn index(&self, idx: [usize; S.rank()]) -> &Self::Output {
        let i = S.storage_idx(&idx).unwrap();
        self.storage.index(i)
    }
}

impl<T: Numeric, const S: Shape, U: ToPrimitive> From<[U; S.num_elems()]> for GenericTensor<T, S> {
    fn from(arr: [U; S.num_elems()]) -> Self {
        arr.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn test_from_iterator() {
        let xs: [i64; 3] = [1, 2, 3];
        let iter = xs.iter().cycle().copied();

        let t1: GenericTensor<f64, { Shape::Rank0([]) }> = iter.clone().collect();
        assert_eq!(t1, GenericTensor::<f64, { Shape::Rank0([]) }>::from([1.0]));

        let t2: GenericTensor<f64, { Shape::Rank2([4, 2]) }> = iter.clone().collect();
        assert_eq!(
            t2,
            GenericTensor::<f64, { Shape::Rank2([4, 2]) }>::from([
                1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0
            ])
        );

        let t3: GenericTensor<f64, { Shape::Rank2([4, 2]) }> = xs.iter().copied().collect();
        assert_eq!(
            t3,
            GenericTensor::<f64, { Shape::Rank2([4, 2]) }>::from([
                1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ])
        );
    }

    #[test]
    #[allow(clippy::zero_prefixed_literal)]
    fn test_from_fn() {
        let f = |idx| {
            let [i, j, k] = idx;
            let s = format!("{}{}{}", i, j, k);
            s.parse().unwrap()
        };
        let t1: GenericTensor<f64, { Shape::Rank3([2; 3]) }> = GenericTensor::from_fn(f);
        assert_eq!(
            t1,
            GenericTensor::<f64, { Shape::Rank3([2; 3]) }>::from([
                000, 001, 010, 011, 100, 101, 110, 111
            ]),
        );

        let t2: GenericTensor<f64, { Shape::Rank3([1, 2, 3]) }> = GenericTensor::from_fn(f);
        assert_eq!(
            t2,
            GenericTensor::<f64, { Shape::Rank3([1, 2, 3]) }>::from([000, 001, 002, 010, 011, 012]),
        );

        let t3: GenericTensor<f64, { Shape::Rank3([3, 2, 1]) }> = GenericTensor::from_fn(f);
        assert_eq!(
            t3,
            GenericTensor::<f64, { Shape::Rank3([3, 2, 1]) }>::from([000, 010, 100, 110, 200, 210]),
        );

        let t4: GenericTensor<f64, { Shape::Rank3([2, 3, 1]) }> = GenericTensor::from_fn(f);
        assert_eq!(
            t4,
            GenericTensor::<f64, { Shape::Rank3([2, 3, 1]) }>::from([000, 010, 020, 100, 110, 120]),
        );
    }

    #[test]
    fn test_math() {
        let mut x: GenericTensor<f64, { Shape::Rank3([1, 2, 2]) }> =
            GenericTensor::from([1, 2, 3, 4]);
        let y: GenericTensor<f64, { Shape::Rank3([1, 2, 2]) }> = GenericTensor::from([5, 6, 7, 8]);
        let a: GenericTensor<f64, { Shape::Rank3([1, 2, 2]) }> =
            GenericTensor::from([6, 8, 10, 12]);

        assert_eq!(x.clone() + &y, a);

        x = x + &y;
        assert_eq!(x.clone(), a);

        let b: GenericTensor<f64, { Shape::Rank3([1, 2, 2]) }> =
            GenericTensor::from([12, 16, 20, 24]);
        assert_eq!(x.clone() * 2.0, b);

        x = x * 2.0;
        assert_eq!(x, b);
    }

    #[test]
    fn test_to_iter() {
        let t: GenericTensor<f64, { Shape::Rank2([2; 2]) }> = (0..4).collect();
        let vals: Vec<f64> = t.into_iter().collect();
        assert_eq!(vals, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_map() {
        let t: GenericTensor<f64, { Shape::Rank2([2; 2]) }> = GenericTensor::from([1, 2, 3, 4]);
        let u = t.map(&|val| val * 2.0);

        let want: GenericTensor<f64, { Shape::Rank2([2; 2]) }> = GenericTensor::from([2, 4, 6, 8]);
        assert_eq!(u, want);
    }

    #[test]
    fn test_reshape() {
        let t = GenericTensor::<f64, { Shape::Rank2([3, 2]) }>::from([1, 2, 3, 4, 5, 6]);
        let t2 = t.clone().reshape::<{ Shape::Rank2([2, 3]) }>();
        let t3 = t.clone().reshape::<{ Shape::Rank1([6]) }>();

        assert_eq!(t[[1, 0]], t2[[0, 2]]);
        assert_eq!(t[[2, 1]], t3[[5]]);
    }
}
