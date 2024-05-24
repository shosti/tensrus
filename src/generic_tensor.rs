use crate::numeric::Numeric;
use crate::scalar::Scalar;
use crate::slice::Slice;
use crate::tensor::{
    downrank, num_elems, transpose_shape, BasicTensor, IndexError, Shape, SlicedTensor, Tensor,
    TensorIterator, Transpose,
};
use crate::type_assert::{Assert, IsTrue};
use num::ToPrimitive;
use std::any::Any;
use std::ops::{Add, Index, Mul};

#[derive(Debug, Clone)]
pub struct GenericTensor<T: Numeric, const R: usize, const S: Shape> {
    pub(crate) storage: Vec<T>,
    pub transpose: Transpose,
}

// Returns the tensor shape when downranking by 1
pub const fn subtensor_shape(r: usize, s: Shape) -> Shape {
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

impl<T: Numeric, const R: usize, const S: Shape> GenericTensor<T, R, S> {
    pub fn new(storage: Vec<T>, transpose: Transpose) -> Self {
        Self { storage, transpose }
    }

    fn storage_size() -> usize {
        num_elems(R, S)
    }

    fn storage_idx(&self, idx: &[usize; R]) -> Result<usize, IndexError> {
        if R == 0 {
            return Ok(0);
        }
        for (dim, &cur) in idx.iter().enumerate() {
            if cur >= S[dim] {
                return Err(IndexError {});
            }
        }

        match self.transpose {
            Transpose::None => Ok(Self::calc_storage_idx(idx, S)),
            Transpose::Transposed => {
                let orig_shape = transpose_shape(R, S);
                let mut orig_idx = *idx;
                orig_idx.reverse();

                Ok(Self::calc_storage_idx(&orig_idx, orig_shape))
            }
        }
    }

    fn calc_storage_idx(idx: &[usize; R], shape: Shape) -> usize {
        let stride = crate::tensor::stride(R, shape);
        let mut i = 0;
        for (dim, &cur) in idx.iter().enumerate() {
            i += stride[dim] * cur;
        }

        i
    }

    pub fn reshape<const R2: usize, const S2: Shape>(self) -> GenericTensor<T, R2, S2>
    where
        Assert<{ num_elems(R, S) == num_elems(R2, S2) }>: IsTrue,
    {
        GenericTensor {
            storage: self.storage,
            transpose: self.transpose,
        }
    }

    pub fn transpose(self) -> GenericTensor<T, R, { transpose_shape(R, S) }>
    where
        Assert<{ R >= 2 }>: IsTrue,
    {
        GenericTensor::new(self.storage, self.transpose.transpose())
    }

    pub(crate) fn is_transposed(&self) -> bool {
        self.transpose == Transpose::Transposed
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
                self.storage[self.storage_idx(&self_idx).unwrap()]
            });
        Ok(out)
    }
}

impl<T: Numeric, const R: usize, const S: Shape> Tensor for GenericTensor<T, R, S> {
    type T = T;
    type Idx = [usize; R];

    fn num_elems() -> usize {
        Self::storage_size()
    }

    fn set(self, idx: &[usize; R], val: T) -> Self {
        match self.storage_idx(idx) {
            Ok(i) => {
                let mut storage = self.storage;
                storage[i] = val;
                Self {
                    storage,
                    transpose: self.transpose,
                }
            }
            Err(_e) => panic!("set: out of bounds"),
        }
    }

    fn map(mut self, f: impl Fn(&Self::Idx, Self::T) -> Self::T) -> Self {
        let mut next_idx = Some(Self::default_idx());
        while let Some(idx) = next_idx {
            let i = self.storage_idx(&idx).unwrap();
            self.storage[i] = f(&idx, self.storage[i]);
            next_idx = self.next_idx(&idx);
        }

        Self {
            storage: self.storage,
            transpose: self.transpose,
        }
    }

    fn default_idx() -> Self::Idx {
        [0; R]
    }
    fn next_idx(&self, idx: &Self::Idx) -> Option<Self::Idx> {
        if R == 0 {
            return None;
        }

        let mut cur = *idx;
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
        Self {
            storage,
            transpose: Transpose::default(),
        }
    }
}

impl<T: Numeric, const R: usize, const S: Shape> BasicTensor<T> for GenericTensor<T, R, S> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_boxed(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    fn num_elems(&self) -> usize {
        Self::storage_size()
    }

    fn clone_boxed(&self) -> Box<dyn BasicTensor<T>> {
        Box::new(self.clone())
    }

    fn zeros_with_shape(&self) -> Box<dyn BasicTensor<T>> {
        Box::new(Self::zeros())
    }

    fn ones_with_shape(&self) -> Box<dyn BasicTensor<T>> {
        Box::new(Self::ones())
    }

    // Scales `self` by `scale` and adds to `other`
    fn add(self: Box<Self>, other_basic: &dyn BasicTensor<T>, scale: T) -> Box<dyn BasicTensor<T>> {
        let other = Self::ref_from_basic(other_basic);
        let out = (*self * scale) + other;
        Box::new(out)
    }
}

impl<T: Numeric, const R: usize, const S: Shape> Index<&[usize; R]> for GenericTensor<T, R, S> {
    type Output = T;

    fn index(&self, idx: &[usize; R]) -> &Self::Output {
        self.storage.index(self.storage_idx(idx).unwrap())
    }
}

impl<T: Numeric, const R: usize, const S: Shape> Index<&[usize]> for GenericTensor<T, R, S> {
    type Output = T;

    fn index(&self, idx: &[usize]) -> &Self::Output {
        if idx.len() != R {
            panic!("invalid index for tensor of rank {}", R);
        }
        let mut i = [0; R];
        i.copy_from_slice(idx);
        self.index(&i)
    }
}

impl<T: Numeric, const R: usize, const S: Shape> SlicedTensor<T, R, S> for GenericTensor<T, R, S> {
    fn try_slice<const D: usize>(
        &self,
        idx: [usize; D],
    ) -> Result<Slice<T, { R - D }, { downrank(R, S, D) }>, IndexError> {
        Slice::new::<D, R, S>(&self.storage, self.transpose, idx)
    }
}

impl<T: Numeric, const R: usize, const S: Shape, F> From<[F; num_elems(R, S)]>
    for GenericTensor<T, R, S>
where
    F: ToPrimitive,
{
    fn from(arr: [F; num_elems(R, S)]) -> Self {
        arr.into_iter().collect()
    }
}

impl<T: Numeric, const R: usize, const S: Shape, F> FromIterator<F> for GenericTensor<T, R, S>
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
        Self {
            storage: vals,
            transpose: Transpose::default(),
        }
    }
}

impl<T: Numeric, const R: usize, const S: Shape> PartialEq for GenericTensor<T, R, S> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().all(|(idx, val)| val == other[&idx])
    }
}

impl<'a, T: Numeric, const R: usize, const S: Shape> IntoIterator for &'a GenericTensor<T, R, S> {
    type Item = ([usize; R], T);
    type IntoIter = TensorIterator<'a, GenericTensor<T, R, S>>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter::new(self)
    }
}

impl<T: Numeric, const R: usize, const S: Shape> Eq for GenericTensor<T, R, S> {}

impl<'a, T: Numeric, const R: usize, const S: Shape> Add<&'a Self> for GenericTensor<T, R, S> {
    type Output = Self;

    fn add(self, other: &Self) -> Self::Output {
        self.map(|idx, v| v + other[idx])
    }
}

impl<T: Numeric, const R: usize, const S: Shape> Mul<T> for GenericTensor<T, R, S> {
    type Output = Self;

    fn mul(self, other: T) -> Self::Output {
        self.map(|_, v| v * other)
    }
}

impl<T: Numeric, const R: usize, const S: Shape> Mul<Scalar<T>> for GenericTensor<T, R, S> {
    type Output = Self;

    fn mul(self, other: Scalar<T>) -> Self::Output {
        self * other.val()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn test_equal() {
        let a: GenericTensor<f64, 0, { [0; 5] }> = GenericTensor::from([1]);
        let b: GenericTensor<f64, 0, { [0; 5] }> = GenericTensor::from([2]);
        assert_ne!(a, b);

        let x: GenericTensor<f64, 2, { [2; 5] }> = GenericTensor::from([1, 2, 3, 4]);
        let y: GenericTensor<f64, 2, { [2; 5] }> = GenericTensor::from([1, 2, 3, 5]);
        assert_ne!(x, y);
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
        let f = |idx: &[usize; 3]| {
            let [i, j, k] = *idx;
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
        let vals: Vec<f64> = t.into_iter().values().collect();
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

    fn test_get_and_set<const R: usize, const S: Shape>(t: GenericTensor<f64, R, S>) {
        let mut rng = rand::thread_rng();
        let mut x = t;
        for _ in 0..10 {
            let mut idx = [0; R];
            for (dim, cur) in idx.iter_mut().enumerate() {
                *cur = rng.gen_range(0..S[dim]);
            }
            let val: f64 = rng.gen();
            x = x.set(&idx, val);

            assert_eq!(x[&idx], val);
        }
    }

    #[test]
    fn test_map() {
        let t: GenericTensor<f64, 2, { [2; 5] }> = GenericTensor::from([1, 2, 3, 4]);
        let u = t.map(|_, val| val * 2.0);

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
    #[rustfmt::skip]
    fn test_transpose() {
        let t: GenericTensor<f64, 2, { [3, 2, 0, 0, 0] }> = GenericTensor::from([
            1, 2,
            3, 4,
            5, 6,
        ]);
        let t2 = t.transpose().map(|_, v| v + 1.0);
        let want: GenericTensor<f64, 2, { [2, 3, 0, 0, 0] }> = GenericTensor::from([
            2, 4, 6,
            3, 5, 7,
        ]);

        assert_eq!(t2, want);
    }

    #[test]
    fn test_reshape() {
        let t = GenericTensor::<f64, 2, { [3, 2, 0, 0, 0] }>::from([1, 2, 3, 4, 5, 6]);
        let t2 = t.clone().reshape::<2, { [2, 3, 0, 0, 0] }>();
        let t3 = t.clone().reshape::<1, { [6, 0, 0, 0, 0] }>();

        assert_eq!(t[&[1, 0]], t2[&[0, 2]]);
        assert_eq!(t[&[2, 1]], t3[&[5]]);
    }
}
