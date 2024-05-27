use crate::{
    numeric::Numeric,
    shape::Shape,
    storage::{Layout, Storage},
    tensor2::Tensor2,
};
use num::ToPrimitive;

pub const fn scalar_shape() -> Shape {
    [0; 6]
}

#[derive(Tensor2, Debug, Clone)]
#[tensor_rank = 0]
#[tensor_shape = "scalar_shape()"]
pub struct Scalar2<T: Numeric> {
    storage: Storage<T>,
    layout: Layout,
}

impl<T: Numeric> Scalar2<T> {
    pub fn val(&self) -> T {
        self[&[]]
    }
}

impl<T: Numeric, F> From<F> for Scalar2<T>
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
    use crate::vector2::Vector2;

    #[test]
    fn basics() {
        let a: Scalar2<f64> = Scalar2::from(42);

        assert_eq!(a[&[]], 42.0);
        assert_eq!(a.val(), 42.0);
    }

    #[test]
    fn multiplication() {
        assert_eq!(
            Scalar2::<f64>::from(6) * Scalar2::<f64>::from(7),
            Scalar2::<f64>::from(42)
        );

        let x: Vector2<f64, _> = Vector2::from([1, 2, 3]);
        let a: Scalar2<f64> = Scalar2::from(6);

        assert_eq!(x * a, Vector2::<f64, 3>::from([6, 12, 18]));
    }
}
