use crate::generic_tensor::GenericTensor;
use crate::numeric::Numeric;
use crate::tensor::{Tensor, TensorShape, IndexError};

pub const fn vector_shape(n: usize) -> TensorShape {
    [n; 5]
}

pub type Vector<T, const N: usize> = VectorTensor<T, 1, { vector_shape(N) }>;

#[derive(Tensor, PartialEq, Debug)]
pub struct VectorTensor<T: Numeric, const R: usize, const S: TensorShape>(GenericTensor<T, R, S>);

// #[derive(Debug)]
// pub struct Vector<T: Numeric, const N: usize> {
//     vals: Rc<RefCell<Vec<T>>>,
// }

// impl<T: Numeric, const N: usize> Vector<T, N> {
//     pub fn dot(&self, other: &Self) -> T {
//         let mut res = T::zero();
//         for i in 0..N {
//             res = res + (self.get([i]).unwrap() * other.get([i]).unwrap());
//         }

//         res
//     }
// }

// impl<T: Numeric, const N: usize> From<[T; N]> for Vector<T, N> {
//     fn from(vals: [T; N]) -> Self {
//         Self {
//             vals: Rc::new(RefCell::new(Vec::from(vals))),
//         }
//     }
// }

// impl<const N: usize> From<[i32; N]> for Vector<f32, N> {
//     fn from(vals: [i32; N]) -> Self {
//         let cast: [f32; N] = std::array::from_fn(|i| vals[i] as f32);
//         Self::from(cast)
//     }
// }

// impl<T: Numeric, const N: usize> Tensor<T, 1> for Vector<T, N> {
//     type Transpose = RowVector<T, N>;

//     fn from_fn<F>(mut cb: F) -> Self
//     where
//         F: FnMut([usize; 1]) -> T,
//     {
//         let vals = Rc::new(RefCell::new((0..N).map(|idx| cb([idx])).collect()));
//         Self { vals }
//     }

//     fn shape(&self) -> [usize; 1] {
//         [N]
//     }

//     fn get(&self, idx: [usize; 1]) -> Result<T, IndexError> {
//         let [i] = idx;
//         if i >= N {
//             return Err(IndexError {});
//         }

//         Ok(self.vals.borrow()[i])
//     }

//     fn set(&mut self, idx: [usize; 1], val: T) -> Result<(), IndexError> {
//         let [i] = idx;
//         if i >= N {
//             return Err(IndexError {});
//         }
//         self.vals.borrow_mut()[i] = val;

//         Ok(())
//     }

//     fn transpose(&self) -> Self::Transpose {
//         Self::Transpose {
//             vals: self.vals.clone(),
//         }
//     }

//     fn next_idx(&self, idx: [usize; 1]) -> Option<[usize; 1]> {
//         let [i] = idx;
//         if i + 1 >= N {
//             None
//         } else {
//             Some([i + 1])
//         }
//     }
// }

// impl<T: Numeric, const N: usize> MulAssign<T> for Vector<T, N> {
//     fn mul_assign(&mut self, other: T) {
//         self.update(|n| n * other);
//     }
// }

// impl<T: Numeric, const N: usize> PartialEq for Vector<T, N> {
//     fn eq(&self, other: &Self) -> bool {
//         (0..N).all(|i| self.get([i]) == other.get([i]))
//     }
// }

// impl<T: Numeric, const N: usize> Eq for Vector<T, N> {}

// #[derive(Debug)]
// pub struct RowVector<T: Numeric, const N: usize> {
//     vals: Rc<RefCell<Vec<T>>>,
// }

// impl<T: Numeric, const N: usize> From<[T; N]> for RowVector<T, N> {
//     fn from(vals: [T; N]) -> Self {
//         Vector::from(vals).transpose()
//     }
// }

// impl<const N: usize> From<[i32; N]> for RowVector<f32, N> {
//     fn from(vals: [i32; N]) -> Self {
//         Vector::from(vals).transpose()
//     }
// }

// impl<T: Numeric, const N: usize> Tensor<T, 1> for RowVector<T, N> {
//     type Transpose = Vector<T, N>;

//     fn from_fn<F>(cb: F) -> Self
//     where
//         F: FnMut([usize; 1]) -> T,
//     {
//         Vector::from_fn(cb).transpose()
//     }

//     fn shape(&self) -> [usize; 1] {
//         [N]
//     }

//     fn get(&self, idx: [usize; 1]) -> Result<T, IndexError> {
//         let [i] = idx;
//         if i >= N {
//             return Err(IndexError {});
//         }

//         Ok(self.vals.borrow()[i])
//     }

//     fn set(&mut self, idx: [usize; 1], val: T) -> Result<(), IndexError> {
//         let [i] = idx;
//         if i >= N {
//             return Err(IndexError {});
//         }
//         self.vals.borrow_mut()[i] = val;

//         Ok(())
//     }

//     fn transpose(&self) -> Self::Transpose {
//         Self::Transpose {
//             vals: self.vals.clone(),
//         }
//     }

//     fn next_idx(&self, idx: [usize; 1]) -> Option<[usize; 1]> {
//         let [i] = idx;
//         if i + 1 >= N {
//             None
//         } else {
//             Some([i + 1])
//         }
//     }
// }

// impl<T: Numeric, const N: usize> MulAssign<T> for RowVector<T, N> {
//     fn mul_assign(&mut self, other: T) {
//         self.update(|n| n * other);
//     }
// }

// impl<T: Numeric, const N: usize> PartialEq for RowVector<T, N> {
//     fn eq(&self, other: &Self) -> bool {
//         (0..N).all(|i| self.get([i]) == other.get([i]))
//     }
// }

// impl<T: Numeric, const M: usize, const N: usize> Mul<Matrix<T, M, N>> for RowVector<T, M> {
//     type Output = RowVector<T, N>;

//     fn mul(self, other: Matrix<T, M, N>) -> Self::Output {
//         RowVector::from_fn(|idx| {
//             let [j] = idx;
//             self.transpose().dot(&other.col(j).unwrap())
//         })
//     }
// }

// impl<T: Numeric, const N: usize> Eq for RowVector<T, N> {}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn basics() {
//         let a = Vector::from([1, 2, 3, 4, 5]);

//         assert_eq!(a.shape(), [5]);
//         assert_eq!(a.get([3]), Ok(4.0));
//         assert_eq!(a.get([5]), Err(IndexError {}));
//     }

//         #[test]
//         fn mul_assign() {
//             let mut a = Vector::from([2, 4, 6, 8]);
//             a *= 3.0;

//             assert_eq!(a, Vector::from([6, 12, 18, 24]));
//         }

//         #[test]
//         fn from_fn() {
//             let a: Vector<_, 4> = Vector::from_fn(|idx| idx[0] as f32 * 2.0);

//             assert_eq!(a, Vector::from([0, 2, 4, 6]));
//         }

//         #[test]
//         fn dot_product() {
//             let a = Vector::from([1, 2, 3]);
//             let b = Vector::from([4, 5, 6]);

//             assert_eq!(a.dot(&b), 32.0);
//         }

//         #[test]
//         fn transpose() {
//             let x = Vector::from([1, 2, 3]);

//             assert_eq!(x.transpose(), RowVector::from([1, 2, 3]));
//         }

//         #[test]
//         fn row_vec_matrix_mul() {
//             let x = RowVector::from([1, 2, 3]);
//             let a = Matrix::from([[2, 4], [3, 6], [7, 8]]);

//             assert_eq!(x * a, RowVector::from([29, 40]));
//         }
// }
