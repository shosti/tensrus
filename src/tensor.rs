use crate::numeric::Numeric;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug, PartialEq)]
pub struct IndexError {}

pub type TensorShape = [usize; 5];

pub struct Tensor<T: Numeric, const R: usize, const S: TensorShape> {
    storage: Rc<RefCell<Vec<T>>>,
}

impl<T: Numeric, const R: usize, const S: TensorShape> Tensor<T, R, S> {
    pub fn zeros() -> Self {
        let vals = vec![T::zero(); Self::storage_size()];
        Self {
            storage: Rc::new(RefCell::new(vals)),
        }
    }

    pub fn rank(&self) -> usize {
        R
    }
    pub fn shape(&self) -> [usize; R] {
        let mut s = [0; R];
        for i in 0..R {
            s[i] = S[i];
        }

        s
    }

    pub fn get(&self, idx: &[usize; R]) -> Result<T, IndexError> {
        match Self::storage_idx(idx) {
            Ok(i) => Ok(self.storage.borrow()[i]),
            Err(e) => Err(e),
        }
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
        let mut size = 1;
        for i in 0..R {
            size *= S[i];
        }
        size
    }

    fn storage_idx(idx: &[usize; R]) -> Result<usize, IndexError> {
        if R == 0 {
            return Ok(0);
        }

        let mut i = idx[0];
        for dim in 1..R {
            let dim_idx = idx[dim];
            if dim_idx >= S[dim] {
                return Err(IndexError {});
            }
            i += idx[dim] * S[dim - 1];
        }

        Ok(i)
    }
}

// pub trait Tensor<T: Numeric, const R: usize>: MulAssign<T> + Eq {
//     type Transpose;

//     fn from_fn<F>(cb: F) -> Self
//     where
//         F: FnMut([usize; R]) -> T;
//     fn zeros() -> Self
//     where
//         Self: Sized,
//     {
//         Self::from_fn(|_| T::zero())
//     }

//     fn rank(&self) -> usize {
//         R
//     }
//     fn shape(&self) -> [usize; R];
//     fn get(&self, idx: [usize; R]) -> Result<T, IndexError>;
//     fn set(&mut self, idx: [usize; R], val: T) -> Result<(), IndexError>;
//     fn transpose(&self) -> Self::Transpose;
//     fn next_idx(&self, idx: [usize; R]) -> Option<[usize; R]>;

//     fn update<F>(&mut self, mut cb: F)
//     where
//         F: FnMut(T) -> T,
//     {
//         let mut iter = Some([0; R]);
//         while let Some(idx) = iter {
//             let cur = self.get(idx).unwrap();
//             self.set(idx, cb(cur)).unwrap();
//             iter = self.next_idx(idx);
//         }
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn rank_and_shape() {
        let scalar: Tensor<f64, 0, { [0; 5] }> = Tensor::zeros();

        assert_eq!(scalar.rank(), 0);
        assert_eq!(scalar.shape(), []);

        let vector: Tensor<f64, 1, { [7; 5] }> = Tensor::zeros();

        assert_eq!(vector.rank(), 1);
        assert_eq!(vector.shape(), [7]);

        let matrix: Tensor<f64, 2, { [3, 2, 0, 0, 0] }> = Tensor::zeros();

        assert_eq!(matrix.rank(), 2);
        assert_eq!(matrix.shape(), [3, 2]);

        let tensor3: Tensor<f64, 3, { [7, 8, 2, 0, 0] }> = Tensor::zeros();
        assert_eq!(tensor3.rank(), 3);
        assert_eq!(tensor3.shape(), [7, 8, 2]);
    }

    #[test]
    fn get_and_set() {
        test_get_and_set(Tensor::<f64, 0, { [0; 5] }>::zeros());
        test_get_and_set(Tensor::<f64, 1, { [24; 5] }>::zeros());
        test_get_and_set(Tensor::<f64, 2, { [8, 72, 0, 0, 0] }>::zeros());
        test_get_and_set(Tensor::<f64, 3, { [243, 62, 101, 0, 0] }>::zeros());
        test_get_and_set(Tensor::<f64, 4, { [1, 99, 232, 8, 0] }>::zeros());
    }

    fn test_get_and_set<const R: usize, const S: TensorShape>(t: Tensor<f64, R, S>) {
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
