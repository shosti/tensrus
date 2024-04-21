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

// pub struct Scalar<T: Numeric>(GenericTensor<T, 0, { [0; 5] }>);

// pub struct Vector<T: Numeric, const N: usize>(GenericTensor<T, 1, { vector_shape(N) }>)
// where
//     [(); num_elems(1, vector_shape(N))]:;



// #[cfg(test)]
// mod tests {
//     use super::*;
//     use rand::prelude::*;

//     #[test]
//     fn rank_and_shape() {
//         // let scalar: Scalar<f64> = Scalar::zeros();

//         // assert_eq!(scalar.rank(), 0);
//         // assert_eq!(scalar.shape(), []);

//         // let vector: Vector<f64, 7> = Vector::zeros();

//         // assert_eq!(vector.rank(), 1);
//         // assert_eq!(vector.shape(), [7]);

//         // let matrix: Matrix<f64, 3, 2> = Matrix::zeros();

//         // assert_eq!(matrix.rank(), 2);
//         // assert_eq!(matrix.shape(), [3, 2]);

//         let tensor3: GenericTensor<f64, 3, { [7, 8, 2, 0, 0] }> = GenericTensor::zeros();
//         assert_eq!(tensor3.rank(), 3);
//         assert_eq!(tensor3.shape(), [7, 8, 2]);
//     }

//     #[test]
//     fn from_vec() {}

//     #[test]
//     fn get_and_set() {
//         test_get_and_set(GenericTensor::<f64, 0, { [0; 5] }>::zeros());
//         test_get_and_set(GenericTensor::<f64, 1, { [24; 5] }>::zeros());
//         test_get_and_set(GenericTensor::<f64, 2, { [8, 72, 0, 0, 0] }>::zeros());
//         test_get_and_set(GenericTensor::<f64, 3, { [243, 62, 101, 0, 0] }>::zeros());
//         test_get_and_set(GenericTensor::<f64, 4, { [1, 99, 232, 8, 0] }>::zeros());
//     }

//     fn test_get_and_set<const R: usize, const S: TensorShape>(t: GenericTensor<f64, R, S>) {
//         let mut rng = rand::thread_rng();
//         for _ in 0..10 {
//             let mut idx = [0; R];
//             for dim in 0..R {
//                 idx[dim] = rng.gen_range(0..S[dim]);
//             }
//             let val: f64 = rng.gen();

//             t.set(&idx, val).unwrap();
//             assert_eq!(t.get(&idx).unwrap(), val);
//         }
//     }

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
