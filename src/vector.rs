use crate::{
    blas::BLASOps,
    differentiable::{Differentiable, DifferentiableTensor},
    encoding::EncodingError,
    generic_tensor::{GenericTensor, IntoGeneric},
    matrix::{self, IntoMatrix, Matrix},
    shape::{self, broadcast_compat, Broadcastable, Reducible, Shape, Shaped, Transposable},
    storage::{Layout, Storage, TensorLayout, TensorStorage},
    tensor::{Indexable, Tensor},
    type_assert::{Assert, IsTrue},
    view::View,
};
use std::ops::Index;

pub const RANK: usize = 1;

#[derive(TensorStorage, OwnedTensorStorage, Tensor, Debug, Clone)]
pub struct Vector<T, const N: usize> {
    storage: Storage<T>,
}

impl<T, const N: usize> Vector<T, N> {
    pub const fn shape() -> Shape {
        shape::rank1([N])
    }

    pub fn dot(&self, rhs: &Self) -> T
    where
        T: BLASOps,
    {
        // Safety: dimensions are checked by type params
        unsafe { T::dot(N as i32, &self.storage.data, 1, &rhs.storage.data, 1) }
    }

    pub fn one_hot<U>(n: U) -> Result<Self, EncodingError>
    where
        T: num::One + num::Zero,
        U: num::Integer + num::ToPrimitive,
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

    pub fn as_col_vector(&self) -> View<Matrix<T, N, 1>> {
        self.view().as_matrix()
    }

    pub fn as_row_vector(&self) -> View<Matrix<T, 1, N>> {
        self.as_col_vector().transpose()
    }
}

// Default Vectors as being interpreted as column vectors (seems mathematically
// reasonable)
impl<T, const N: usize> IntoMatrix<T, N, 1> for Vector<T, N> {}

impl<T, const N: usize> Shaped for Vector<T, N> {
    fn rank() -> usize {
        RANK
    }

    fn shape() -> Shape {
        Self::shape()
    }
}

impl<T, const M: usize> Vector<T, M>
where
    T: BLASOps + num::One,
{
    /// add_matvecmul is a typesafe wrapper around gemv implementations that
    /// multiplies `lhs` and `rhs` and adds the result to `self`. Generally,
    /// higher-level Mul operations should be used instead of this method.
    pub fn add_matvecmul<const N: usize>(
        self,
        lhs: View<Matrix<T, M, N>>,
        rhs: View<Vector<T, N>>,
    ) -> Self {
        matvecmul_impl::<T, M, N>(
            &lhs.storage().data,
            lhs.layout(),
            &rhs.storage().data,
            rhs.layout(),
            self,
        )
    }
}

fn matvecmul_impl<T, const M: usize, const N: usize>(
    lhs_data: &[T],
    lhs_layout: Layout,
    rhs_data: &[T],
    rhs_layout: Layout,
    mut out: Vector<T, M>,
) -> Vector<T, M>
where
    T: BLASOps + num::One,
{
    debug_assert!(!out.storage.layout.is_transposed());
    debug_assert!(!rhs_layout.is_transposed());

    // BLAS always uses column-major format, so if we're "transposed" we're
    // already in BLAS format, otherwise we have to transpose.
    let trans = lhs_layout.to_blas();
    let m = if lhs_layout.is_transposed() { M } else { N } as i32;
    let n = if lhs_layout.is_transposed() { N } else { M } as i32;
    let lda = m;

    // Safety: dimensions are checked by type parameters
    unsafe {
        T::gemv(
            trans,
            m,
            n,
            T::one(),
            lhs_data,
            lda,
            rhs_data,
            1,
            T::one(),
            &mut out.storage.data,
            1,
        );
    }

    out
}

impl<T, const N: usize> From<[T; N]> for Vector<T, N> {
    fn from(vals: [T; N]) -> Self {
        Self {
            storage: vals.into(),
        }
    }
}

impl<T, const N: usize> IntoGeneric<T, { RANK }, { Self::shape() }> for Vector<T, N> {}

impl<T, const N: usize> Index<&[usize; RANK]> for Vector<T, N> {
    type Output = T;

    fn index(&self, idx: &[usize; RANK]) -> &Self::Output {
        self.storage
            .index(idx, Self::rank(), Self::shape())
            .unwrap()
    }
}

impl<T, const N: usize> Index<&[usize; RANK]> for &Vector<T, N> {
    type Output = T;

    fn index(&self, idx: &[usize; RANK]) -> &Self::Output {
        (*self).index(idx)
    }
}

impl<T, const N: usize> Indexable for Vector<T, N> {
    type Idx = [usize; 1];
    type T = T;
}

impl<T, const N: usize> DifferentiableTensor for Vector<T, N> where T: Differentiable {}

impl<T, const N: usize, const M: usize, const P: usize> Broadcastable<Matrix<T, M, P>>
    for Vector<T, N>
where
    Assert<
        {
            broadcast_compat(
                RANK,
                Self::shape(),
                matrix::RANK,
                Matrix::<T, M, P>::shape(),
            )
        },
    >: IsTrue,
{
}

impl<T, const N: usize, const R: usize, const S: Shape> Broadcastable<GenericTensor<T, R, S>>
    for Vector<T, N>
where
    Assert<{ broadcast_compat(RANK, Self::shape(), matrix::RANK, S) }>: IsTrue,
{
}

impl<T, const N: usize> Reducible<0> for Vector<T, N> {
    type Reduced = Vector<T, 1>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;

    #[test]
    fn test_basics() {
        let a = Vector::from([1, 2, 3, 4, 5]);

        assert_eq!(a[&[3]], 4);
    }

    #[test]
    fn test_from_fn() {
        let a: Vector<_, 4> = Vector::from_fn(|idx| idx[0] * 2);

        assert_eq!(a, Vector::from([0, 2, 4, 6]));
    }

    #[test]
    fn test_dot_product() {
        let a = Vector::from([1.0, 2.0, 3.0]);
        let b = Vector::from([4.0, 5.0, 6.0]);

        assert_eq!(a.dot(&b), 32.0);
    }

    #[test]
    fn test_one_hot() {
        assert_eq!(
            Vector::<f64, 5>::one_hot(3).unwrap(),
            Vector::<f64, _>::from([0.0, 0.0, 0.0, 1.0, 0.0]),
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
        let x = Vector::from([1, 2, 3]);
        let m1: Matrix<_, 2, 3> = x.view().broadcast().to_owned();
        assert_eq!(m1, Matrix::from([[1, 2, 3], [1, 2, 3]]));

        let m2: Matrix<_, 3, 2> = x.as_col_vector().broadcast().to_owned();
        assert_eq!(m2, Matrix::from([[1, 1], [2, 2], [3, 3]]));
    }
}
