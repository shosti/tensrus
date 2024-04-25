use crate::numeric::Numeric;
use crate::tensor::{num_elems, IndexError, ShapeError, Tensor, TensorShape};
use num::ToPrimitive;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug)]
pub struct GenericTensor<T: Numeric, const R: usize, const S: TensorShape> {
    storage: Rc<RefCell<Vec<T>>>,
}

impl<T: Numeric, const R: usize, const S: TensorShape> GenericTensor<T, R, S> {
    pub fn zeros() -> Self {
        let vals = vec![T::zero(); Self::storage_size()];
        Self {
            storage: Rc::new(RefCell::new(vals)),
        }
    }

    pub fn shape(&self) -> [usize; R] {
        let mut s = [0; R];
        for i in 0..R {
            s[i] = S[i];
        }

        s
    }
    pub fn set(&self, idx: &[usize; R], val: T) -> Result<(), IndexError> {
        match Self::storage_idx(&idx) {
            Ok(i) => {
                let mut storage = self.storage.borrow_mut();
                storage[i] = val;
                Ok(())
            }
            Err(e) => Err(e),
        }
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
        &self,
    ) -> Result<GenericTensor<T, R2, S2>, ShapeError> {
        if num_elems(R, S) != num_elems(R2, S2) {
            return Err(ShapeError {});
        }

        Ok(GenericTensor {
            storage: self.storage.clone(),
        })
    }
}

impl<T: Numeric, const R: usize, const S: TensorShape> Tensor<T, R, S> for GenericTensor<T, R, S> {
    fn from_fn<F>(mut cb: F) -> Self
    where
        F: FnMut([usize; R]) -> T,
    {
        (0..Self::storage_size()).into_iter().map(|i| cb(Self::idx_from_storage_idx(i).unwrap())).collect()
    }

    fn shape(&self) -> [usize; R] {
        let mut s = [0; R];
        for i in 0..R {
            s[i] = S[i];
        }

        s
    }

    fn get(&self, idx: &[usize; R]) -> Result<T, IndexError> {
        match Self::storage_idx(idx) {
            Ok(i) => Ok(self.storage.borrow()[i]),
            Err(e) => Err(e),
        }
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

impl<T: Numeric, const R: usize, const S: TensorShape> Eq for GenericTensor<T, R, S> {}

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
            for dim in 0..R {
                idx[dim] = rng.gen_range(0..S[dim]);
            }
            let val: f64 = rng.gen();

            t.set(&idx, val).unwrap();
            assert_eq!(t.get(&idx).unwrap(), val);
        }
    }
}
