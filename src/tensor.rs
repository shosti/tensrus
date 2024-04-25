use crate::numeric::Numeric;

#[derive(Debug, PartialEq)]
pub struct IndexError {}

#[derive(Debug, PartialEq)]
pub struct ShapeError {}

pub type TensorShape = [usize; 5];

pub const fn vector_shape(n: usize) -> TensorShape {
    [n; 5]
}

pub const fn num_elems(r: usize, s: TensorShape) -> usize {
    let mut dim = 0;
    let mut n = 1;

    while dim < r {
        n *= s[dim];
        dim += 1;
    }

    n
}

pub const fn shape_dim(s: TensorShape, i: usize) -> usize {
    s[i]
}

pub trait Tensor<T: Numeric, const R: usize, const S: TensorShape> {
    fn from_fn<F>(cb: F) -> Self
    where
        F: FnMut([usize; R]) -> T;

    fn rank(&self) -> usize {
        R
    }
    fn shape(&self) -> [usize; R];
    fn get(&self, idx: &[usize; R]) -> Result<T, IndexError>;
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use rand::prelude::*;
//     #[test]
//     fn from_vec() {}

//     #[test]
//     // #[test]
//     // fn vector_basics() {
//     //     let a: Vector<f64, 5> = Vector::from([1.0, 2.0, 3.0, 4.0, 5.0]);

//     //     assert_eq!(a.shape(), [5]);
//     //     assert_eq!(a.get(&[3]), Ok(4.0));
//     //     assert_eq!(a.get(&[5]), Err(IndexError {}));
//     // }

//     // #[test]
//     // fn reshape() {
//     //     let x: Matrix<f64, 3, 2> = Matrix::from([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
//     //     let y: Matrix<f64, 2, 3> = Matrix::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
//     //     let x_reshaped: Matrix<f64, 2, 3> = x.reshape().unwrap();

//     //     assert_eq!(x_reshaped, y);

//     //     let bad_reshape: Result<Matrix<f64, 2, 4>, ShapeError> = x.reshape();
//     //     assert_eq!(bad_reshape, Err(ShapeError {}));

//     //     let vec: Vector<f64, 4> = Vector::from([2.0; 4]);
//     //     let matrix: Matrix<f64, 4, 1> = vec.reshape().unwrap();
//     //     let expected: Matrix<f64, 4, 1> = Matrix::from([2.0; 4]);

//     //     assert_eq!(matrix, expected);
//     // }
// }
