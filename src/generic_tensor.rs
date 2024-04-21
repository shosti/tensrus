use crate::numeric::Numeric;
use crate::tensor::{num_elems, IndexError, ShapeError, Tensor, TensorShape};
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

    // fn from_fn<F>(mut cb: F) -> Self
    // where
    //     F: FnMut([usize; R]) -> T,
    // {
    //     let vals = vec![T::zer(); Self::storage_size()];
    // }
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

    //     fn idx_for_storage_idx(idx: usize) -> Result<[usize; R], IndexError> {

    // //             (0..(M * N)).map(|idx| cb([idx / N, idx % N])).collect(),
    //     }

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

impl<T: Numeric, const R: usize, const S: TensorShape> From<[T; num_elems(R, S)]>
    for GenericTensor<T, R, S>
{
    fn from(arr: [T; num_elems(R, S)]) -> Self {
        let vals: Vec<T> = arr.into_iter().collect();
        Self {
            storage: Rc::new(RefCell::new(vals)),
        }
    }
}

impl<T: Numeric, const R: usize, const S: TensorShape> FromIterator<T> for GenericTensor<T, R, S> {
    fn from_iter<A>(iter: A) -> Self
    where
        A: IntoIterator<Item = T>,
    {
        let vals: Vec<T> = iter
            .into_iter()
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
