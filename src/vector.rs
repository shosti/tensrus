use crate::{
    generic_tensor::GenericTensor,
    matrix::MatrixView,
    numeric::Numeric,
    shape::Shape,
    storage::{Layout, Storage},
    tensor::Tensor,
};
use num::{Integer, ToPrimitive};

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

#[derive(Debug, PartialEq)]
pub enum EncodingError {
    InvalidOneHotInput,
}

impl<T: Numeric, const N: usize> Vector<T, N> {
    pub fn one_hot<U>(n: U) -> Result<Self, EncodingError>
    where
        U: Integer + ToPrimitive,
    {
        let i = n.to_usize().ok_or(EncodingError::InvalidOneHotInput)?;
        if i >= N {
            return Err(EncodingError::InvalidOneHotInput);
        }

        Ok(Self::from_fn(|idx| {
            if idx[0] == i {
                T::one()
            } else {
                T::zero()
            }
        }))
    }

    pub fn dot(&self, other: &Self) -> T {
        unsafe { T::dot(N as i32, &self.storage, 1, &other.storage, 1) }
    }

    pub fn normalize(self) -> NormalizedVector<T, N> {
        let s = self.sum().val();
        if s == T::one() {
            NormalizedVector { v: self }
        } else if s == T::zero() {
            let val = T::one() / T::from(N).unwrap();
            NormalizedVector {
                v: self.map(|_, _| val),
            }
        } else {
            NormalizedVector {
                v: self.map(|_, n| n / s),
            }
        }
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

#[derive(Debug)]
pub struct NormalizedVector<T: Numeric, const N: usize> {
    v: Vector<T, N>,
}

impl<T: Numeric, const N: usize> NormalizedVector<T, N> {
    pub fn as_vector(&self) -> &Vector<T, N> {
        &self.v
    }
}

impl<T: Numeric, const N: usize> std::ops::Index<&[usize; 1]> for NormalizedVector<T, N> {
    type Output = T;

    fn index(&self, index: &[usize; 1]) -> &Self::Output {
        self.v.index(index)
    }
}

impl<T: Numeric, const N: usize> From<NormalizedVector<T, N>> for Vector<T, N> {
    fn from(v: NormalizedVector<T, N>) -> Self {
        v.v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;
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

    #[test]
    fn test_one_hot() {
        assert_eq!(
            Vector::<f64, 5>::one_hot(3).unwrap(),
            Vector::<f64, _>::from([0, 0, 0, 1, 0]),
        );
        assert_eq!(
            Vector::<f64, 5>::one_hot(-3),
            Err(EncodingError::InvalidOneHotInput),
        );
        assert_eq!(
            Vector::<f64, 5>::one_hot(7),
            Err(EncodingError::InvalidOneHotInput),
        );
    }

    #[test]
    fn test_broadcast() {
        let x: Vector<f64, _> = [1, 2, 3].into();
        let m1: Matrix<f64, 2, 3> = x.view().to_generic().broadcast().from_generic().into();
        assert_eq!(m1, Matrix::<f64, _, _>::from([[1, 2, 3], [1, 2, 3]]));

        let x_col = x.as_col_vector().view();
        let m2: Matrix<f64, 3, 2> = x_col.to_generic().broadcast().from_generic().into();
        assert_eq!(m2, Matrix::<f64, _, _>::from([[1, 1], [2, 2], [3, 3]]));
    }

    seq!(N in 1..10 {
        proptest! {
            #[test]
            fn test_normalize_~N(v in prop::collection::vec(any::<f64>(), N)) {
                let x: Vector<f64, N> = v.into_iter().collect();
                let x_n = x.normalize();

                const TOLERANCE: f64 = 0.00001;
                assert!((x_n.as_vector().sum().val() - 1.0).abs() < TOLERANCE);
            }
        }
    });
}
