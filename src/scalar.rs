use crate::generic_tensor::GenericTensor;
use crate::numeric::Numeric;
use crate::tensor::{IndexError, Tensor, TensorIterator, TensorShape};
use num::ToPrimitive;
use std::ops::Mul;

pub const fn scalar_shape() -> TensorShape {
    [0; 5]
}

#[derive(Tensor, PartialEq, Debug)]
pub struct ScalarTensor<T: Numeric, const R: usize, const S: TensorShape>(GenericTensor<T, R, S>);

pub type Scalar<T> = ScalarTensor<T, 0, { scalar_shape() }>;

impl<T: Numeric> Scalar<T> {
    pub fn val(&self) -> T {
        self.get(&[]).unwrap()
    }
}

impl<T: Numeric, F> From<F> for Scalar<T>
where
    F: ToPrimitive + Copy,
{
    fn from(val: F) -> Self {
        let t: GenericTensor<T, 0, { scalar_shape() }> = std::iter::once(val).collect();
        Self(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::Vector;

    #[test]
    fn basics() {
        let a: Scalar<f64> = Scalar::from(42);

        assert_eq!(a.shape(), []);
        assert_eq!(a.get(&[]), Ok(42.0));
        assert_eq!(a.val(), 42.0);
    }

    #[test]
    fn multiplication() {
        assert_eq!(
            Scalar::<f64>::from(6) * Scalar::<f64>::from(7),
            Scalar::<f64>::from(42)
        );

        let x: Vector<f64, _> = Vector::from([1, 2, 3]);
        let a: Scalar<f64> = Scalar::from(6);

        assert_eq!(x * a, Vector::<f64, 3>::from([6, 12, 18]));
    }
}
