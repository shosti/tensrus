use crate::{
    distribution::Multinomial, generic_tensor::GenericTensor, matrix::MatrixView, numeric::Numeric, shape::Shape, storage::{Layout, Storage}, tensor::Tensor
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

// Convenience types for rows/columns
type RowVector<'a, T, const N: usize> = MatrixView<'a, T, 1, N>;
type ColumnVector<'a, T, const N: usize> = MatrixView<'a, T, N, 1>;

impl<T: Numeric, const N: usize> Vector<T, N> {
    pub fn dot(&self, other: &Self) -> T {
        unsafe { T::dot(N as i32, &self.storage, 1, &other.storage, 1) }
    }

    // Normalizes vector so that entries sum to 0
    pub fn normalize(self) -> Self {
        let s = self.sum().val();
        if s == T::zero() {
            return self.map(|_, _| T::one() / T::from(N).unwrap());
        }
        self.map(|_, n| n / s)
    }

    pub fn to_multinomial(self) -> Multinomial<T, N> {
        self.into()
    }

    pub fn as_col_vector(&self) -> ColumnVector<T, N> {
        debug_assert_eq!(self.layout, Layout::Normal);

        MatrixView {
            storage: &self.storage,
            layout: self.layout,
        }
    }

    pub fn as_row_vector(&self) -> RowVector<T, N> {
        self.as_col_vector().transpose()
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
    use proptest::prelude::*;
    use seq_macro::seq;

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

    seq!(N in 1..10 {
        proptest! {
            #[test]
            fn test_normalize_~N(v in prop::collection::vec(any::<f64>(), N)) {
                let x: Vector<f64, N> = v.into_iter().collect();
                let x_n = x.normalize();

                const TOLERANCE: f64 = 0.00001;
                assert!((x_n.sum().val() - 1.0).abs() < TOLERANCE);
            }
        }
    });
}
