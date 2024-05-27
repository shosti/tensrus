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
