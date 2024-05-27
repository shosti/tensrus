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
    pub(crate) storage: Storage<T>,
    pub layout: Layout,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basics() {
        let a: Vector2<f64, 5> = Vector2::from([1, 2, 3, 4, 5]);

        assert_eq!(a[&[3]], 4.0);
    }

    #[test]
    fn test_from_fn() {
        let a: Vector2<_, 4> = Vector2::from_fn(|idx| idx[0] as f32 * 2.0);

        assert_eq!(a, Vector2::from([0, 2, 4, 6]));
    }

    #[test]
    fn test_dot_product() {
        let a: Vector2<f64, _> = Vector2::from([1, 2, 3]);
        let b: Vector2<f64, _> = Vector2::from([4, 5, 6]);

        assert_eq!(a.dot(&b), 32.0);
    }
}
