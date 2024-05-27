use crate::{
    generic_tensor::GenericTensor,
    numeric::Numeric,
    shape::Shape,
    storage::{Layout, Storage},
    tensor::Tensor,
};
use num::ToPrimitive;

pub const fn vector_shape(n: usize) -> Shape {
    [n, 0, 0, 0, 0, 0]
}

#[derive(Tensor, Debug, Clone)]
#[tensor_rank = 1]
#[tensor_shape = "vector_shape(N)"]
pub struct Vector<T: Numeric, const N: usize> {
    pub(crate) storage: Storage<T>,
    pub layout: Layout,
}

impl<T: Numeric, const N: usize> Vector<T, N> {
    pub fn dot(&self, other: &Self) -> T {
        unsafe { T::dot(N as i32, &self.storage, 1, &other.storage, 1) }
    }
}

impl<T: Numeric, const N: usize, F: ToPrimitive> From<[F; N]> for Vector<T, N> {
    fn from(arr: [F; N]) -> Self {
        let vals = arr.into_iter().map(|v| T::from(v).unwrap()).collect();
        Self {
            storage: vals,
            layout: Layout::default(),
        }
    }
}

impl<T: Numeric, const N: usize> From<GenericTensor<T, 1, { vector_shape(N) }>> for Vector<T, N> {
    fn from(t: GenericTensor<T, 1, { vector_shape(N) }>) -> Self {
        Self {
            storage: t.storage,
            layout: t.layout,
        }
    }
}

impl<T: Numeric, const N: usize> From<Vector<T, N>> for GenericTensor<T, 1, { vector_shape(N) }> {
    fn from(t: Vector<T, N>) -> Self {
        Self {
            storage: t.storage,
            layout: t.layout,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basics() {
        let a: Vector<f64, 5> = Vector::from([1, 2, 3, 4, 5]);

        assert_eq!(a[&[3]], 4.0);
    }

    #[test]
    fn test_from_fn() {
        let a: Vector<_, 4> = Vector::from_fn(|idx| idx[0] as f32 * 2.0);

        assert_eq!(a, Vector::from([0, 2, 4, 6]));
    }

    #[test]
    fn test_dot_product() {
        let a: Vector<f64, _> = Vector::from([1, 2, 3]);
        let b: Vector<f64, _> = Vector::from([4, 5, 6]);

        assert_eq!(a.dot(&b), 32.0);
    }
}
