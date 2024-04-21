use crate::numeric::Numeric;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug, PartialEq)]
pub struct IndexError {}

type TensorShape = [usize; 5];

pub struct Tensor<T: Numeric, const R: usize, const S: TensorShape> {
    storage: Rc<RefCell<Vec<T>>>,
}

// Useful type definitions for scalar/vector/matrix
pub type Scalar<T> = Tensor<T, 0, { [0; 5] }>;

pub const fn vector_shape(n: usize) -> TensorShape {
    [n; 5]
}

pub type Vector<T, const N: usize> = Tensor<T, 1, { vector_shape(N) }>;

pub const fn matrix_shape(m: usize, n: usize) -> TensorShape {
    [m, n, 0, 0, 0]
}

pub type Matrix<T, const M: usize, const N: usize> = Tensor<T, 2, { matrix_shape(M, N) }>;

const fn num_elems(r: usize, s: TensorShape) -> usize {
    let mut dim = 0;
    let mut n = 1;

    while dim < r {
        n *= s[dim];
        dim += 1;
    }

    n
}

impl<T: Numeric, const R: usize, const S: TensorShape> Tensor<T, R, S> {
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
}

const fn shape_dim(s: TensorShape, i: usize) -> usize {
    s[i]
}

impl<T: Numeric, const R: usize, const S: TensorShape> From<[T; num_elems(R, S)]>
    for Tensor<T, R, S>
{
    fn from(arr: [T; num_elems(R, S)]) -> Self {
        let vals: Vec<T> = arr.into_iter().collect();
        Self {
            storage: Rc::new(RefCell::new(vals)),
        }
    }
}

impl<T: Numeric, const S: TensorShape> From<[[T; shape_dim(S, 1)]; shape_dim(S, 0)]>
    for Tensor<T, 2, S>
{
    fn from(arrs: [[T; shape_dim(S, 1)]; shape_dim(S, 0)]) -> Self {
        let vals: Vec<T> = arrs.into_iter().flatten().collect();

        Self {
            storage: Rc::new(RefCell::new(vals)),
        }
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
        let scalar: Scalar<f64> = Scalar::zeros();

        assert_eq!(scalar.rank(), 0);
        assert_eq!(scalar.shape(), []);

        let vector: Vector<f64, 7> = Vector::zeros();

        assert_eq!(vector.rank(), 1);
        assert_eq!(vector.shape(), [7]);

        let matrix: Matrix<f64, 3, 2> = Matrix::zeros();

        assert_eq!(matrix.rank(), 2);
        assert_eq!(matrix.shape(), [3, 2]);

        let tensor3: Tensor<f64, 3, { [7, 8, 2, 0, 0] }> = Tensor::zeros();
        assert_eq!(tensor3.rank(), 3);
        assert_eq!(tensor3.shape(), [7, 8, 2]);
    }

    #[test]
    fn from_vec() {}

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

    #[test]
    fn matrix_basics() {
        let x: Matrix<f64, 4, 3> = Matrix::from([
            [3.0, 4.0, 5.0],
            [2.0, 7.0, 9.0],
            [6.0, 5.0, 10.0],
            [3.0, 7.0, 3.0],
        ]);

        assert_eq!(x.shape(), [4, 3]);
        assert_eq!(x.get(&[2, 1]), Ok(5.0));
        assert_eq!(x.get(&[3, 2]), Ok(3.0));
        assert_eq!(x.get(&[4, 1]), Err(IndexError {}));
    }

    #[test]
    fn vector_basics() {
        let a: Vector<f64, 5> = Vector::from([1.0, 2.0, 3.0, 4.0, 5.0]);

        assert_eq!(a.shape(), [5]);
        assert_eq!(a.get(&[3]), Ok(4.0));
        assert_eq!(a.get(&[5]), Err(IndexError {}));
    }
}
