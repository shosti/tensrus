use crate::generic_tensor::GenericTensor;
use crate::numeric::Numeric;
use crate::tensor::{num_elems, IndexError, Tensor, TensorShape};
use num::ToPrimitive;

pub const fn scalar_shape() -> TensorShape {
    [0; 5]
}

#[derive(Tensor, PartialEq, Debug)]
pub struct ScalarTensor<T: Numeric, const R: usize, const S: TensorShape>(GenericTensor<T, R, S>);

pub type Scalar<T> = ScalarTensor<T, 0, { scalar_shape() }>;

impl<T: Numeric, const R: usize, const S: TensorShape, F> From<F> for ScalarTensor<T, R, S>
where
    F: ToPrimitive + Copy,
    [(); num_elems(R, S)]:,
{
    fn from(val: F) -> Self {
        let t: GenericTensor<T, R, S> = GenericTensor::from([val; num_elems(R, S)]);
        Self(t)
    }
}

// #[derive(Debug)]
// pub struct Scalar<T: Numeric> {
//     val: T,
// }

// impl<T: Numeric> From<T> for Scalar<T> {
//     fn from(val: T) -> Self {
//         Scalar { val }
//     }
// }

// impl<T: Numeric> Tensor<T, 0> for Scalar<T> {
//     type Transpose = Self;

//     fn from_fn<F>(mut cb: F) -> Self
//     where
//         F: FnMut([usize; 0]) -> T,
//     {
//         Self { val: cb([]) }
//     }

//     fn shape(&self) -> [usize; 0] {
//         []
//     }

//     fn get(&self, _idx: [usize; 0]) -> Result<T, IndexError> {
//         Ok(self.val)
//     }

//     fn set(&mut self, _idx: [usize; 0], val: T) -> Result<(), IndexError> {
//         self.val = val;
//         Ok(())
//     }

//     fn transpose(&self) -> Self::Transpose {
//         Self::Transpose { val: self.val }
//     }

//     fn next_idx(&self, _idx: [usize; 0]) -> Option<[usize; 0]> {
//         None
//     }
// }

// impl<T: Numeric> PartialEq for Scalar<T> {
//     fn eq(&self, other: &Self) -> bool {
//         self.get([]) == other.get([])
//     }
// }

// impl<T: Numeric> Eq for Scalar<T> {}

// impl<T: Numeric> MulAssign<T> for Scalar<T> {
//     fn mul_assign(&mut self, other: T) {
//         self.update(|n| n * other);
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basics() {
        let a: Scalar<f64> = Scalar::from(42);

        assert_eq!(a.shape(), []);
        assert_eq!(a.get(&[]), Ok(42.0));
    }
}
