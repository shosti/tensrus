use crate::{
    numeric::Numeric,
    shape::Shape,
    storage::{Layout, Storage},
    tensor::Tensor,
};
use num::ToPrimitive;

pub const fn scalar_shape() -> Shape {
    [0; 6]
}

#[derive(Tensor, Debug, Clone)]
#[tensor_rank = 0]
#[tensor_shape = "scalar_shape()"]
pub struct Scalar<T: Numeric> {
    storage: Storage<T>,
    layout: Layout,
}

impl<T: Numeric> Scalar<T> {
    pub fn val(&self) -> T {
        self[&[]]
    }
}

impl<T: Numeric, F> From<F> for Scalar<T>
where
    F: ToPrimitive + Copy,
{
    fn from(val: F) -> Self {
        let vals = vec![T::from(val).unwrap()];
        Self {
            storage: vals.into(),
            layout: Layout::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::Vector;

    #[test]
    fn basics() {
        let a: Scalar<f64> = Scalar::from(42);

        assert_eq!(a[&[]], 42.0);
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
