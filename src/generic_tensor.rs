use crate::numeric::Numeric;
use crate::scalar::Scalar;
use crate::tensor::{
    num_elems, BasicTensor, IndexError, ShapeError, ShapedTensor, Tensor, TensorIterator,
    TensorShape,
};
use num::ToPrimitive;
use std::any::Any;
use std::cell::RefCell;
use std::ops::{Add, AddAssign, Mul, MulAssign};
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct GenericTensor<T: Numeric, const R: usize, const S: TensorShape> {
    storage: Rc<RefCell<Vec<T>>>,
}

impl<T: Numeric, const R: usize, const S: TensorShape> GenericTensor<T, R, S> {
    fn storage_size() -> usize {
        num_elems(R, S)
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
}

impl<T: Numeric, const R: usize, const S: TensorShape> BasicTensor for GenericTensor<T, R, S> {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl<T: Numeric, const R: usize, const S: TensorShape> ShapedTensor<R, S>
    for GenericTensor<T, R, S>
{
}

impl<T: Numeric, const R: usize, const S: TensorShape> Tensor for GenericTensor<T, R, S> {
    type T = T;
    type Idx = [usize; R];

    fn get(&self, idx: [usize; R]) -> T {
        match Self::storage_idx(idx) {
            Ok(i) => self.storage.borrow()[i],
            Err(_e) => panic!("get: out of bounds"),
        }
    }

    fn set(&self, idx: [usize; R], val: T) {
        match Self::storage_idx(idx) {
            Ok(i) => {
                let mut storage = self.storage.borrow_mut();
                storage[i] = val;
            }
            Err(_e) => panic!("set: out of bounds"),
        }
    }

    fn zeros() -> Self {
        let storage = vec![T::zero(); Self::storage_size()];
        Self {
            storage: Rc::new(RefCell::new(storage)),
        }
    }

    fn update<F: Fn(T) -> T>(&mut self, f: F) {
        self.storage
            .borrow_mut()
            .iter_mut()
            .for_each(|v| *v = f(*v));
    }

    fn update_zip<F: Fn(T, T) -> T>(&mut self, other: &Self, f: F) {
        self.storage
            .borrow_mut()
            .iter_mut()
            .zip(other.storage.borrow().iter())
            .for_each(|(x, y)| *x = f(*x, *y))
    }

    fn update_zip2<F: Fn(T, T, T) -> T>(&mut self, a: &Self, b: &Self, f: F) {
        self.storage
            .borrow_mut()
            .iter_mut()
            .zip(a.storage.borrow().iter())
            .zip(b.storage.borrow().iter())
            .for_each(|((x, y), z)| *x = f(*x, *y, *z))
    }

    fn deep_clone(&self) -> Self {
        Self {
            storage: Rc::new(RefCell::new(self.storage.borrow().clone())),
        }
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

    fn from_fn<F>(cb: F) -> Self
    where
        F: Fn([usize; R]) -> T,
    {
        (0..Self::storage_size())
            .map(|i| cb(Self::idx_from_storage_idx(i).unwrap()))
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
        Self {
            storage: Rc::new(RefCell::new(vals)),
        }
    }
}

impl<T: Numeric, const R: usize, const S: TensorShape> PartialEq for GenericTensor<T, R, S> {
    fn eq(&self, other: &Self) -> bool {
        for i in 0..Self::storage_size() {
            // TODO: use helpers once they're there
            if self.storage.borrow()[i] != other.storage.borrow()[i] {
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

impl<T: Numeric, const R: usize, const S: TensorShape> Add for GenericTensor<T, R, S> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self::from_fn(|idx| self.get(idx) + other.get(idx))
    }
}

impl<T: Numeric, const R: usize, const S: TensorShape> AddAssign for GenericTensor<T, R, S> {
    fn add_assign(&mut self, other: Self) {
        self.update_zip(&other, |x, y| x + y)
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

impl<T: Numeric, const R: usize, const S: TensorShape> MulAssign<T> for GenericTensor<T, R, S> {
    fn mul_assign(&mut self, other: T) {
        self.update(&|v| v * other);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn basics() {
        assert_eq!(GenericTensor::<f64, 2, { [2, 5, 0, 0, 0] }>::stride(), [5, 1]);
        assert_eq!(GenericTensor::<f64, 3, { [2, 3, 3, 0, 0] }>::stride(), [9, 3, 1]);
    }

    #[test]
    fn from_iterator() {
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
    fn math() {
        let mut x: GenericTensor<f64, 3, { [1, 2, 2, 0, 0] }> = GenericTensor::from([1, 2, 3, 4]);
        let y: GenericTensor<f64, 3, { [1, 2, 2, 0, 0] }> = GenericTensor::from([5, 6, 7, 8]);
        let a: GenericTensor<f64, 3, { [1, 2, 2, 0, 0] }> = GenericTensor::from([6, 8, 10, 12]);

        assert_eq!(x.clone() + y.clone(), a);

        x += y;
        assert_eq!(x.clone(), a);

        let b: GenericTensor<f64, 3, { [1, 2, 2, 0, 0] }> = GenericTensor::from([12, 16, 20, 24]);
        assert_eq!(x.clone() * 2.0, b);

        x *= 2.0;
        assert_eq!(x, b);
    }

    #[test]
    fn clone() {
        let mut x: GenericTensor<f64, 1, { [2; 5] }> = GenericTensor::from([1, 2]);
        let y: GenericTensor<f64, 1, { [2; 5] }> = GenericTensor::from([1, 2]);
        let z: GenericTensor<f64, 1, { [2; 5] }> = GenericTensor::from([2, 4]);

        let x_clone = x.clone();
        assert_eq!(x_clone, y);

        x *= 2.0;
        assert_eq!(x, z);
        assert_eq!(x_clone, z);

        let mut deep = x.deep_clone();
        deep *= 0.5;

        assert_eq!(deep, y);
        assert_eq!(x, z);
    }

    #[test]
    fn to_iter() {
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
        for _ in 0..10 {
            let mut idx = [0; R];
            for (dim, cur) in idx.iter_mut().enumerate() {
                *cur = rng.gen_range(0..S[dim]);
            }
            let val: f64 = rng.gen();

            t.set(idx, val);
            assert_eq!(t.get(idx), val);
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
