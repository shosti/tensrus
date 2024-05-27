use crate::{
    numeric::Numeric,
    shape::Shape,
    storage::{Layout, Storage},
    tensor2::Tensor2,
};
use num::ToPrimitive;

pub const fn vector_shape(n: usize) -> Shape {
    [n, 0, 0, 0, 0, 0]
}

#[derive(Tensor2, Debug, Clone)]
#[tensor_rank = 1]
#[tensor_shape = "vector_shape(N)"]
pub struct Vector2<T: Numeric, const N: usize> {
    storage: Storage<T>,
    layout: Layout,
}

impl<T: Numeric, const N: usize> Vector2<T, N> {
    pub fn dot(&self, other: &Self) -> T {
        unsafe { T::dot(N as i32, &self.storage, 1, &other.storage, 1) }
    }
}

impl<T: Numeric, const N: usize, F: ToPrimitive> From<[F; N]> for Vector2<T, N> {
    fn from(arr: [F; N]) -> Self {
        let vals = arr.into_iter().map(|v| T::from(v).unwrap()).collect();
        Self {
            storage: vals,
            layout: Layout::default(),
        }
    }
}
