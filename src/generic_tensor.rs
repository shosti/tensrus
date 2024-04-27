use crate::numeric::Numeric;
use crate::scalar::Scalar;
use crate::tensor::{
    num_elems, IndexError, ShapeError, Tensor, TensorIterator, TensorOps, TensorShape,
};
use num::ToPrimitive;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign};

#[derive(Debug)]
pub struct GenericTensor<T: Numeric, const R: usize, const S: TensorShape> {
    storage: Vec<T>,
}

impl<T: Numeric, const R: usize, const S: TensorShape> GenericTensor<T, R, S> {
    pub fn zeros() -> Self {
        let storage = vec![T::zero(); Self::storage_size()];
        Self { storage }
    }

    fn storage_size() -> usize {
        num_elems(R, S)
    }

    fn idx_from_storage_idx(idx: usize) -> Result<[usize; R], IndexError> {
        if idx >= Self::storage_size() {
            return Err(IndexError {});
        }

        let mut res = [0; R];
        let mut i = idx;

        for dim in 0..R {
            let offset: usize = S[(dim + 1)..R].iter().product();
            let cur = i / offset;
            res[dim] = cur;
            i -= cur * offset;
        }
        debug_assert!(i == 0);
        debug_assert!(Self::storage_idx(&res).unwrap() == idx);

        Ok(res)
    }

    fn storage_idx(idx: &[usize; R]) -> Result<usize, IndexError> {
        if R == 0 {
            return Ok(0);
        }

        let mut i = 0;
        for dim in 0..R {
            if idx[dim] >= S[dim] {
                return Err(IndexError {});
            }
            let offset: usize = S[(dim + 1)..R].iter().product();
            i += offset * idx[dim];
        }

        Ok(i)
    }

    pub fn reshape<const R2: usize, const S2: TensorShape>(
        self,
    ) -> Result<GenericTensor<T, R2, S2>, ShapeError> {
        if num_elems(R, S) != num_elems(R2, S2) {
            return Err(ShapeError {});
        }

        Ok(GenericTensor {
            storage: self.storage,
        })
    }

    pub fn from_fn<F>(cb: F) -> Self
    where
        F: Fn([usize; R]) -> T,
    {
        (0..Self::storage_size())
            .into_iter()
            .map(|i| cb(Self::idx_from_storage_idx(i).unwrap()))
            .collect()
    }
}

impl<T: Numeric, const R: usize, const S: TensorShape> Tensor<T, R, S> for GenericTensor<T, R, S> {
    fn get(&self, idx: &[usize; R]) -> Result<T, IndexError> {
        match Self::storage_idx(idx) {
            Ok(i) => Ok(self.storage[i]),
            Err(e) => Err(e),
        }
    }

    fn set(&mut self, idx: &[usize; R], val: T) -> Result<(), IndexError> {
        match Self::storage_idx(&idx) {
            Ok(i) => {
                self.storage[i] = val;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn update(&mut self, f: &dyn Fn(T) -> T) {
        self.storage.iter_mut().for_each(|v| *v = f(*v));
    }
}

impl<T: Numeric, const R: usize, const S: TensorShape> TensorOps<T> for GenericTensor<T, R, S> {}

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
        let storage: Vec<T> = iter
            .into_iter()
            .map(|v| T::from(v).unwrap())
            .chain(std::iter::repeat(T::zero()))
            .take(Self::storage_size())
            .collect();
        Self { storage }
    }
}

impl<T: Numeric, const R: usize, const S: TensorShape> Clone for GenericTensor<T, R, S> {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
        }
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
    type IntoIter = TensorIterator<'a, T, R, S>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter::new(self)
    }
}

impl<T: Numeric, const R: usize, const S: TensorShape> Eq for GenericTensor<T, R, S> {}

impl<T: Numeric, const R: usize, const S: TensorShape> Add<T> for GenericTensor<T, R, S> {
    type Output = Self;

    fn add(self, other: T) -> Self::Output {
        Self::from_fn(|idx| self.get(&idx).unwrap() + other)
    }
}

impl<T: Numeric, const R: usize, const S: TensorShape> Add<Scalar<T>> for GenericTensor<T, R, S> {
    type Output = Self;

    fn add(self, other: Scalar<T>) -> Self::Output {
        Self::from_fn(|idx| self.get(&idx).unwrap() + other.val())
    }
}

impl<T: Numeric, const R: usize, const S: TensorShape> AddAssign<T> for GenericTensor<T, R, S> {
    fn add_assign(&mut self, other: T) {
        self.update(&|v| v + other);
    }
}

impl<T: Numeric, const R: usize, const S: TensorShape> Mul<T> for GenericTensor<T, R, S> {
    type Output = Self;

    fn mul(self, other: T) -> Self::Output {
        Self::from_fn(|idx| self.get(&idx).unwrap() * other)
    }
}

impl<T: Numeric, const R: usize, const S: TensorShape> Mul<Scalar<T>> for GenericTensor<T, R, S> {
    type Output = Self;

    fn mul(self, other: Scalar<T>) -> Self::Output {
        Self::from_fn(|idx| self.get(&idx).unwrap() * other.val())
    }
}

impl<T: Numeric, const R: usize, const S: TensorShape> MulAssign<T> for GenericTensor<T, R, S> {
    fn mul_assign(&mut self, other: T) {
        self.update(&|v| v * other);
    }
}

impl<T: Numeric, const R: usize, const S: TensorShape> Index<[usize; R]>
    for GenericTensor<T, R, S>
{
    type Output = T;

    fn index(&self, index: [usize; R]) -> &Self::Output {
        self.storage.index(Self::storage_idx(&index).unwrap())
    }
}

impl<T: Numeric, const R: usize, const S: TensorShape> IndexMut<[usize; R]>
    for GenericTensor<T, R, S>
{
    fn index_mut(&mut self, index: [usize; R]) -> &mut Self::Output {
        self.storage.index_mut(Self::storage_idx(&index).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn from_iterator() {
        let xs: [i64; 3] = [1, 2, 3];
        let iter = xs.iter().cycle().map(|x| *x);

        let t1: GenericTensor<f64, 0, { [0; 5] }> = iter.clone().collect();
        assert_eq!(t1, GenericTensor::<f64, 0, { [0; 5] }>::from([1.0]));

        let t2: GenericTensor<f64, 2, { [4, 2, 0, 0, 0] }> = iter.clone().collect();
        assert_eq!(
            t2,
            GenericTensor::<f64, 2, { [4, 2, 0, 0, 0] }>::from([
                1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0
            ])
        );

        let t3: GenericTensor<f64, 2, { [4, 2, 0, 0, 0] }> = xs.iter().map(|x| *x).collect();
        assert_eq!(
            t3,
            GenericTensor::<f64, 2, { [4, 2, 0, 0, 0] }>::from([
                1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ])
        );
    }

    #[test]
    fn from_fn() {
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
    fn to_iter() {
        let t: GenericTensor<f64, 2, { [2; 5] }> = (0..4).into_iter().collect();
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

    fn test_get_and_set<const R: usize, const S: TensorShape>(mut t: GenericTensor<f64, R, S>) {
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            let mut idx = [0; R];
            for dim in 0..R {
                idx[dim] = rng.gen_range(0..S[dim]);
            }
            let val: f64 = rng.gen();

            t.set(&idx, val).unwrap();
            assert_eq!(t.get(&idx).unwrap(), val);
        }
    }

    #[test]
    fn update() {
        let mut t: GenericTensor<f64, 2, { [2; 5] }> = GenericTensor::from([1, 2, 3, 4]);
        t.update(&|val| val * 2.0);

        let want: GenericTensor<f64, 2, { [2; 5] }> = GenericTensor::from([2, 4, 6, 8]);
        assert_eq!(t, want);
    }
}
