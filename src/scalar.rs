use crate::generic_tensor::GenericTensor;
use crate::numeric::Numeric;
use crate::tensor::Shape;
use num::ToPrimitive;

pub const fn scalar_shape() -> Shape {
    [0; 5]
}

#[derive(Tensor, PartialEq, Debug)]
pub struct Scalar<T: Numeric>(GenericTensor<T, 0, { scalar_shape() }>);

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
        let t: GenericTensor<T, 0, { scalar_shape() }> = std::iter::once(val).collect();
        Self(t)
    }
}
