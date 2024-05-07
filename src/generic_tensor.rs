use crate::numeric::Numeric;
use crate::scalar::Scalar;
use crate::shape::{Dims, Shape};
use crate::tensor::{num_elems, IndexError, Tensor, TensorIterator, TensorShape};
use crate::type_assert::{Assert, IsTrue};
use num::ToPrimitive;
use std::ops::{Add, Mul};

pub struct GenericTensor2<T: Numeric, const S: Shape> {
    storage: Vec<T>,
}

impl<T: Numeric, const S: Shape> GenericTensor2<T, S>
where
    Shape: Into<Dims<{ S.rank() }>>,
{
    fn storage_size() -> usize {
        S.len()
    }

    fn stride() -> [usize; S.rank()] {
        let mut res = [0; S.rank()];
        let dims: Dims<{ S.rank() }> = S.into();
        for (dim, item) in res.iter_mut().enumerate() {
            let mut n = 1;
            for d in  (dim + 1)..S.rank() {
                n *= dims[d];
            }
            *item = n;
        }

        res
    }

    fn idx_from_storage_idx(idx: usize) -> Result<[usize; S.rank()], IndexError> {
        if idx >= Self::storage_size() {
            return Err(IndexError {});
        }

        let mut res = [0; S.rank()];
        let mut i = idx;
        let stride = Self::stride();

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

        let mut i = 0;
        let stride = Self::stride();
        let dims: Dims<{ S.rank() }> = S.into();
        for (dim, &cur) in idx.iter().enumerate() {
            if cur >= dims[dim] {
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

#[derive(Debug, Clone)]
pub struct GenericTensor<T: Numeric, const R: usize, const S: TensorShape> {
    pub(crate) storage: Vec<T>,
}

// Returns the tensor shape when downranking by 1
pub const fn subtensor_shape(r: usize, s: TensorShape) -> TensorShape {
    if r == 0 {
        panic!("cannot take subtensor of tensor of rank 0");
    }
    let mut out = [0; 5];
    let mut i = r - 1;
    while i > 0 {
        out[i - 1] = s[i];
        i -= 1;
    }

    out
}

impl<T: Numeric, const R: usize, const S: TensorShape> GenericTensor<T, R, S> {
    fn storage_size() -> usize {
        num_elems(R, S)
    }

    fn stride() -> [usize; R] {
        let mut res = [0; R];
        for (dim, item) in res.iter_mut().enumerate() {
            *item = S[(dim + 1)..R].iter().product();
        }

        res
    }

    fn idx_from_storage_idx(idx: usize) -> Result<[usize; R], IndexError> {
        if idx >= Self::storage_size() {
            return Err(IndexError {});
        }

        let mut res = [0; R];
        let mut i = idx;
        let stride = Self::stride();

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

    fn storage_idx(idx: [usize; R]) -> Result<usize, IndexError> {
        if R == 0 {
            return Ok(0);
        }

        let mut i = 0;
        let stride = Self::stride();
        for (dim, &cur) in idx.iter().enumerate() {
            if cur >= S[dim] {
                return Err(IndexError {});
            }
            i += stride[dim] * idx[dim];
        }

        Ok(i)
    }

    pub fn reshape<const R2: usize, const S2: TensorShape>(self) -> GenericTensor<T, R2, S2>
    where
        Assert<{ num_elems(R, S) == num_elems(R2, S2) }>: IsTrue,
    {
        GenericTensor {
            storage: self.storage,
        }
    }

    pub fn subtensor(
        &self,
        i: usize,
    ) -> Result<GenericTensor<T, { R - 1 }, { subtensor_shape(R, S) }>, IndexError> {
        if i >= S[0] {
            return Err(IndexError {});
        }

        let out: GenericTensor<T, { R - 1 }, { subtensor_shape(R, S) }> =
            GenericTensor::from_fn(|idx| {
                let mut self_idx = [i; R];
                self_idx[1..R].copy_from_slice(&idx[..(R - 1)]);
                self.storage[Self::storage_idx(self_idx).unwrap()]
            });
        Ok(out)
    }
}

impl<T: Numeric, const R: usize, const S: TensorShape> Tensor for GenericTensor<T, R, S> {
    type T = T;
    type Idx = [usize; R];

    fn get(&self, idx: [usize; R]) -> T {
        match Self::storage_idx(idx) {
            Ok(i) => self.storage[i],
            Err(_e) => panic!("get: out of bounds"),
        }
    }

    fn set(self, idx: [usize; R], val: T) -> Self {
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
        [0; R]
    }
    fn next_idx(idx: Self::Idx) -> Option<Self::Idx> {
        let mut cur = idx;
        cur[R - 1] += 1;
        for dim in (0..R).rev() {
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

impl<T: Numeric, const R: usize, const S: TensorShape, F> From<[F; num_elems(R, S)]>
    for GenericTensor<T, R, S>
where
    F: ToPrimitive,
{
    fn from(arr: [F; num_elems(R, S)]) -> Self {
        arr.into_iter().collect()
    }
}

impl<T: Numeric, const R: usize, const S: TensorShape, F> FromIterator<F> for GenericTensor<T, R, S>
where
    F: ToPrimitive,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = F>,
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

impl<T: Numeric, const R: usize, const S: TensorShape> PartialEq for GenericTensor<T, R, S> {
    fn eq(&self, other: &Self) -> bool {
        for i in 0..Self::storage_size() {
            if self.storage[i] != other.storage[i] {
                return false;
            }
        }
        true
    }
}

impl<'a, T: Numeric, const R: usize, const S: TensorShape> IntoIterator
    for &'a GenericTensor<T, R, S>
{
    type Item = T;
    type IntoIter = TensorIterator<'a, GenericTensor<T, R, S>>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter::new(self)
    }
}

impl<T: Numeric, const R: usize, const S: TensorShape> Eq for GenericTensor<T, R, S> {}

impl<'a, T: Numeric, const R: usize, const S: TensorShape> Add<&'a Self>
    for GenericTensor<T, R, S>
{
    type Output = Self;

    fn add(self, other: &Self) -> Self::Output {
        self.zip(other).map(|vs| vs[0] + vs[1])
    }
}

impl<T: Numeric, const R: usize, const S: TensorShape> Mul<T> for GenericTensor<T, R, S> {
    type Output = Self;

    fn mul(self, other: T) -> Self::Output {
        Self::from_fn(|idx| self.get(idx) * other)
    }
}

impl<T: Numeric, const R: usize, const S: TensorShape> Mul<Scalar<T>> for GenericTensor<T, R, S> {
    type Output = Self;

    fn mul(self, other: Scalar<T>) -> Self::Output {
        Self::from_fn(|idx| self.get(idx) * other.val())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn test_basics() {
        assert_eq!(
            GenericTensor::<f64, 2, { [2, 5, 0, 0, 0] }>::stride(),
            [5, 1]
        );
        assert_eq!(
            GenericTensor::<f64, 3, { [2, 3, 3, 0, 0] }>::stride(),
            [9, 3, 1]
        );
    }

    #[test]
    fn test_from_iterator() {
        let xs: [i64; 3] = [1, 2, 3];
        let iter = xs.iter().cycle().copied();

        let t1: GenericTensor<f64, 0, { [0; 5] }> = iter.clone().collect();
        assert_eq!(t1, GenericTensor::<f64, 0, { [0; 5] }>::from([1.0]));

        let t2: GenericTensor<f64, 2, { [4, 2, 0, 0, 0] }> = iter.clone().collect();
        assert_eq!(
            t2,
            GenericTensor::<f64, 2, { [4, 2, 0, 0, 0] }>::from([
                1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0
            ])
        );

        let t3: GenericTensor<f64, 2, { [4, 2, 0, 0, 0] }> = xs.iter().copied().collect();
        assert_eq!(
            t3,
            GenericTensor::<f64, 2, { [4, 2, 0, 0, 0] }>::from([
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
        let t1: GenericTensor<f64, 3, { [2; 5] }> = GenericTensor::from_fn(f);
        assert_eq!(
            t1,
            GenericTensor::<f64, 3, { [2; 5] }>::from([000, 001, 010, 011, 100, 101, 110, 111]),
        );

        let t2: GenericTensor<f64, 3, { [1, 2, 3, 0, 0] }> = GenericTensor::from_fn(f);
        assert_eq!(
            t2,
            GenericTensor::<f64, 3, { [1, 2, 3, 0, 0] }>::from([000, 001, 002, 010, 011, 012]),
        );

        let t3: GenericTensor<f64, 3, { [3, 2, 1, 0, 0] }> = GenericTensor::from_fn(f);
        assert_eq!(
            t3,
            GenericTensor::<f64, 3, { [3, 2, 1, 0, 0] }>::from([000, 010, 100, 110, 200, 210]),
        );

        let t4: GenericTensor<f64, 3, { [2, 3, 1, 0, 0] }> = GenericTensor::from_fn(f);
        assert_eq!(
            t4,
            GenericTensor::<f64, 3, { [2, 3, 1, 0, 0] }>::from([000, 010, 020, 100, 110, 120]),
        );
    }

    #[test]
    fn test_math() {
        let mut x: GenericTensor<f64, 3, { [1, 2, 2, 0, 0] }> = GenericTensor::from([1, 2, 3, 4]);
        let y: GenericTensor<f64, 3, { [1, 2, 2, 0, 0] }> = GenericTensor::from([5, 6, 7, 8]);
        let a: GenericTensor<f64, 3, { [1, 2, 2, 0, 0] }> = GenericTensor::from([6, 8, 10, 12]);

        assert_eq!(x.clone() + &y, a);

        x = x + &y;
        assert_eq!(x.clone(), a);

        let b: GenericTensor<f64, 3, { [1, 2, 2, 0, 0] }> = GenericTensor::from([12, 16, 20, 24]);
        assert_eq!(x.clone() * 2.0, b);

        x = x * 2.0;
        assert_eq!(x, b);
    }

    #[test]
    fn test_to_iter() {
        let t: GenericTensor<f64, 2, { [2; 5] }> = (0..4).collect();
        let vals: Vec<f64> = t.into_iter().collect();
        assert_eq!(vals, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn get_and_set() {
        test_get_and_set(GenericTensor::<f64, 0, { [0; 5] }>::zeros());
        test_get_and_set(GenericTensor::<f64, 1, { [24; 5] }>::zeros());
        test_get_and_set(GenericTensor::<f64, 2, { [8, 72, 0, 0, 0] }>::zeros());
        test_get_and_set(GenericTensor::<f64, 3, { [243, 62, 101, 0, 0] }>::zeros());
        test_get_and_set(GenericTensor::<f64, 4, { [1, 99, 232, 8, 0] }>::zeros());
    }

    fn test_get_and_set<const R: usize, const S: TensorShape>(t: GenericTensor<f64, R, S>) {
        let mut rng = rand::thread_rng();
        let mut x = t;
        for _ in 0..10 {
            let mut idx = [0; R];
            for (dim, cur) in idx.iter_mut().enumerate() {
                *cur = rng.gen_range(0..S[dim]);
            }
            let val: f64 = rng.gen();
            x = x.set(idx, val);

            assert_eq!(x.get(idx), val);
        }
    }

    #[test]
    fn test_map() {
        let t: GenericTensor<f64, 2, { [2; 5] }> = GenericTensor::from([1, 2, 3, 4]);
        let u = t.map(&|val| val * 2.0);

        let want: GenericTensor<f64, 2, { [2; 5] }> = GenericTensor::from([2, 4, 6, 8]);
        assert_eq!(u, want);
    }

    #[test]
    fn test_subtensor() {
        let t3: GenericTensor<f64, 3, { [2, 3, 4, 0, 0] }> = (1..25).collect();

        let t2 = t3.subtensor(1).unwrap();
        let t2_expected: GenericTensor<f64, 2, { [3, 4, 0, 0, 0] }> = (13..25).collect();
        assert_eq!(t2, t2_expected);
        assert_eq!(t3.subtensor(2), Err(IndexError {}));

        let t1 = t2.subtensor(1).unwrap();
        let t1_expected: GenericTensor<f64, 1, { [4, 0, 0, 0, 0] }> =
            GenericTensor::from([17, 18, 19, 20]);
        assert_eq!(t1, t1_expected);

        let t0 = t1.subtensor(1).unwrap();
        let t0_expected: GenericTensor<f64, 0, { [0; 5] }> = GenericTensor::from([18]);
        assert_eq!(t0, t0_expected);
    }

    #[test]
    fn test_reshape() {
        let t = GenericTensor::<f64, 2, { [3, 2, 0, 0, 0] }>::from([1, 2, 3, 4, 5, 6]);
        let t2 = t.clone().reshape::<2, { [2, 3, 0, 0, 0] }>();
        let t3 = t.clone().reshape::<1, { [6, 0, 0, 0, 0] }>();

        assert_eq!(t.get([1, 0]), t2.get([0, 2]));
        assert_eq!(t.get([2, 1]), t3.get([5]));
    }
}
