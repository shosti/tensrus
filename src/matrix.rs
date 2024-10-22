use crate::{
    blas::BLASOps,
    differentiable::{Differentiable, DifferentiableTensor},
    errors::IndexError,
    generic_tensor::{GenericTensor, IntoGeneric},
    shape::{self, broadcast_compat, Broadcastable, Reducible, Shape, Shaped, Transposable},
    storage::{Layout, OwnedTensorStorage, Storage, TensorLayout, TensorStorage},
    tensor::{Indexable, Tensor},
    translation::Translation,
    type_assert::{Assert, IsTrue},
    vector::Vector,
    view::View,
};
use std::{
    fmt::Debug,
    ops::{Index, Mul},
};

pub const RANK: usize = 2;

#[derive(TensorStorage, OwnedTensorStorage, Tensor, Clone)]
pub struct Matrix<T, const M: usize, const N: usize> {
    storage: Storage<T>,
}

pub trait IntoMatrix<T, const M: usize, const N: usize>: OwnedTensorStorage<T> + Sized {
    fn into_matrix(self) -> Matrix<T, M, N> {
        Matrix {
            storage: self.into_storage(),
        }
    }
}

impl<T, const M: usize, const N: usize> Matrix<T, M, N> {
    pub const fn shape() -> Shape {
        shape::rank2([M, N])
    }

    pub fn row(&self, i: usize) -> Result<Translation<Vector<T, N>>, IndexError> {
        if i >= M {
            return Err(IndexError::OutOfBounds);
        }

        let t = self.view().translate(move |idx: [usize; 1]| {
            let j = idx[0];
            [i, j]
        });

        Ok(t)
    }

    pub fn col(&self, j: usize) -> Result<Translation<Vector<T, M>>, IndexError> {
        if j >= N {
            return Err(IndexError::OutOfBounds);
        }

        let v = self.view().translate(move |idx: [usize; 1]| {
            let i = idx[0];
            [i, j]
        });

        Ok(v)
    }

    pub(crate) fn print(
        this: impl Indexable<Idx = <Matrix<T, M, N> as Indexable>::Idx, T = T>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> Result<(), std::fmt::Error>
    where
        T: Debug,
    {
        for row in 0..M {
            for col in 0..N {
                write!(f, "\t{:?}", this[&[row, col]])?;
            }
            writeln!(f)?;
        }

        Ok(())
    }

    pub fn from_matrix<Tn>(self) -> Tn
    where
        Tn: IntoMatrix<T, M, N>,
    {
        Tn::from_storage(self.storage)
    }
}

impl<T, const M: usize, const N: usize> IntoMatrix<T, M, N> for Matrix<T, M, N> {}

impl<T, const N: usize> Matrix<T, N, N>
where
    T: num::One + num::Zero,
{
    pub fn identity() -> Self {
        Self::from_fn(|[i, j]| if i == j { T::one() } else { T::zero() })
    }
}

impl<T, const M: usize, const P: usize> Matrix<T, M, P>
where
    T: BLASOps + num::One,
{
    /// matmul is a typesafe wrapper around gemm implementations that multiplies
    /// `lhs` and `rhs` and adds the result to `self`. Generally, higher-level
    /// Mul operations should be used instead of this method.
    pub fn add_matmul<const N: usize>(
        self,
        lhs: View<Matrix<T, M, N>>,
        rhs: View<Matrix<T, N, P>>,
    ) -> Self {
        matmul_impl::<T, M, N, P>(
            &lhs.storage().data,
            lhs.layout(),
            &rhs.storage().data,
            rhs.layout(),
            self,
        )
    }
}

impl<'a, T, const M: usize, const N: usize, const P: usize> Mul<View<'a, Matrix<T, N, P>>>
    for View<'a, Matrix<T, M, N>>
where
    T: Copy + num::Zero + num::One + BLASOps,
{
    type Output = Matrix<T, M, P>;

    fn mul(self, rhs: View<'a, Matrix<T, N, P>>) -> Self::Output {
        let out = Matrix::zeros();
        out.add_matmul(self, rhs)
    }
}

impl<T, const M: usize, const N: usize, const P: usize> Mul<&Matrix<T, N, P>> for &Matrix<T, M, N>
where
    T: Copy + num::Zero + num::One + BLASOps,
{
    type Output = Matrix<T, M, P>;

    fn mul(self, rhs: &Matrix<T, N, P>) -> Self::Output {
        self.view() * rhs.view()
    }
}

impl<'a, T, const M: usize, const N: usize> Mul<View<'a, Vector<T, N>>>
    for View<'a, Matrix<T, M, N>>
where
    T: Copy + num::Zero + num::One + BLASOps,
{
    type Output = Vector<T, M>;

    fn mul(self, rhs: View<'a, Vector<T, N>>) -> Self::Output {
        let out = Vector::zeros();
        out.add_matvecmul(self, rhs)
    }
}

impl<T, const M: usize, const N: usize> Mul<&Vector<T, N>> for &Matrix<T, M, N>
where
    T: Copy + num::Zero + num::One + BLASOps,
{
    type Output = Vector<T, M>;

    fn mul(self, rhs: &Vector<T, N>) -> Self::Output {
        self.view() * rhs.view()
    }
}

fn matmul_impl<T, const M: usize, const N: usize, const P: usize>(
    lhs_data: &[T],
    lhs_layout: Layout,
    rhs_data: &[T],
    rhs_layout: Layout,
    mut out: Matrix<T, M, P>,
) -> Matrix<T, M, P>
where
    T: BLASOps + num::One,
{
    // We need the out format to be column-major; if it isn't, take (B^T * A*T)^T
    if !out.storage.layout.is_transposed() {
        let out_t = matmul_impl::<T, P, N, M>(
            rhs_data,
            rhs_layout.transpose(),
            lhs_data,
            lhs_layout.transpose(),
            out.transpose(),
        );
        return out_t.transpose();
    }

    debug_assert!(out.storage.layout.is_transposed());

    // Safety: dimensions are enforced by the const params.
    unsafe {
        T::gemm(
            lhs_layout.to_blas(),
            rhs_layout.to_blas(),
            M as i32,
            P as i32,
            N as i32,
            T::one(),
            lhs_data,
            if lhs_layout.is_transposed() { M } else { N } as i32,
            rhs_data,
            if rhs_layout.is_transposed() { N } else { P } as i32,
            T::one(),
            &mut out.storage.data,
            M as i32,
        )
    }

    out
}

impl<T, const M: usize, const N: usize> From<[[T; N]; M]> for Matrix<T, M, N>
where
    T: Default,
{
    fn from(arrs: [[T; N]; M]) -> Self {
        arrs.into_iter().flatten().collect()
    }
}

impl<T, const M: usize, const N: usize> Shaped for Matrix<T, M, N> {
    fn rank() -> usize {
        RANK
    }

    fn shape() -> Shape {
        Self::shape()
    }
}

impl<T, const M: usize, const N: usize> IntoGeneric<T, { RANK }, { Self::shape() }>
    for Matrix<T, M, N>
{
}

impl<T, const M: usize, const N: usize> Index<&[usize; RANK]> for Matrix<T, M, N> {
    type Output = T;

    fn index(&self, idx: &[usize; RANK]) -> &Self::Output {
        self.storage
            .index(idx, Self::rank(), Self::shape())
            .unwrap()
    }
}

impl<T, const M: usize, const N: usize> Index<&[usize; RANK]> for &Matrix<T, M, N> {
    type Output = T;

    fn index(&self, idx: &[usize; RANK]) -> &Self::Output {
        (*self).index(idx)
    }
}

impl<T, const M: usize, const N: usize> Indexable for Matrix<T, M, N> {
    type Idx = [usize; 2];
    type T = T;
}

impl<T, const M: usize, const N: usize> DifferentiableTensor for Matrix<T, M, N> where
    T: Differentiable
{
}

impl<T, const M: usize, const N: usize> Transposable<Matrix<T, N, M>> for Matrix<T, M, N> {
    fn transpose(self) -> Matrix<T, N, M> {
        Matrix {
            storage: self.storage.transpose(),
        }
    }
}

impl<T, const M: usize, const N: usize> Debug for Matrix<T, M, N>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Matrix<{}, {}> [", M, N)?;
        Self::print(self, f)?;
        writeln!(f, "]")?;

        Ok(())
    }
}

impl<T, const M: usize, const N: usize> Reducible<0> for Matrix<T, M, N> {
    type Reduced = Matrix<T, 1, N>;
}
impl<T, const M: usize, const N: usize> Reducible<1> for Matrix<T, M, N> {
    type Reduced = Matrix<T, M, 1>;
}

impl<T, const M: usize, const N: usize, const R_DEST: usize, const S_DEST: Shape>
    Broadcastable<GenericTensor<T, R_DEST, S_DEST>> for Matrix<T, M, N>
where
    Assert<{ broadcast_compat(RANK, Self::shape(), R_DEST, S_DEST) }>: IsTrue,
{
}

impl<T, const M: usize, const N: usize, const P: usize, const U: usize>
    Broadcastable<Matrix<T, P, U>> for Matrix<T, M, N>
where
    Assert<{ broadcast_compat(RANK, Self::shape(), RANK, Matrix::<T, P, U>::shape()) }>: IsTrue,
{
}

#[cfg(test)]
mod tests {
    use crate::{generic_tensor::GenericTensor, view::View};

    use super::*;

    #[test]
    #[rustfmt::skip]
    fn test_matrix_basics() {
        let x: Matrix<f64, _, _> = Matrix::from([
            [3, 4, 5],
            [2, 7, 9],
            [6, 5, 10],
            [3, 7, 3],
        ]).into_iter().map(|(_, val)| val as f64).collect();

        assert_eq!(x[&[2, 1]], 5.0);
        assert_eq!(x[&[3, 2]], 3.0);

        let y: Matrix<f64, 4, 3> = Matrix::from([
            [3.0, 4.0, 5.0],
            [2.0, 7.0, 9.0],
            [6.0, 5.0, 10.0],
            [3.0, 7.0, 3.0]
        ]);
        assert_eq!(x, y);
    }

    #[test]
    fn test_add() {
        let x = Matrix::from([[1, 2], [3, 4]]);
        let y_g = GenericTensor::<_, 2, { Matrix::<i32, 2, 2>::shape() }>::from([1, 2, 3, 4]);
        let y: View<Matrix<_, 2, 2>> = y_g.view().from_generic();
        assert_eq!(x + y, Matrix::from([[2, 4], [6, 8]]));
    }

    #[test]
    #[rustfmt::skip]
    fn test_from_iter() {
        let x: Matrix<f64, 3, 2> = [1.0, 2.0, 3.0].into_iter().cycle().collect();
        let y: Matrix<f64, _, _> = Matrix::from([
            [1.0, 2.0],
            [3.0, 1.0],
            [2.0, 3.0],
        ]);
        assert_eq!(x, y);
    }

    #[test]
    #[allow(clippy::zero_prefixed_literal)]
    fn test_from_fn() {
        let x = Matrix::from_fn(|idx| {
            let [i, j] = idx;
            let s = format!("{}{}", i, j);
            s.parse().unwrap()
        });
        let y = Matrix::from([[00, 01, 02, 03], [10, 11, 12, 13], [20, 21, 22, 23]]);

        assert_eq!(x, y);
    }

    #[test]
    fn test_matrix_multiply() {
        #[rustfmt::skip]
        let x: Matrix<f64, _, _> = Matrix::from([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ]);
        #[rustfmt::skip]
        let x_t: Matrix<f64, _, _> = Matrix::from([
            [1.0, 3.0, 5.0],
            [2.0, 4.0, 6.0],
        ]).transpose();

        #[rustfmt::skip]
        let y: Matrix<f64, _, _> = Matrix::from([
            [7.0, 8.0, 9.0, 10.0],
            [9.0, 10.0, 11.0, 12.0]
        ]);
        #[rustfmt::skip]
        let y_t: Matrix<f64, _, _> = Matrix::from([
            [7.0, 9.0],
            [8.0, 10.0],
            [9.0, 11.0],
            [10.0, 12.0],
        ]).transpose();

        #[rustfmt::skip]
        let want: Matrix<f64, _, _> = Matrix::from([
            [25.0, 28.0, 31.0, 34.0],
            [57.0, 64.0, 71.0, 78.0],
            [89.0, 100.0, 111.0, 122.0]
        ]);

        assert_eq!(&x * &y, want);
        assert_eq!(&x_t * &y, want);
        assert_eq!(&x * &y_t, want);
        assert_eq!(&x_t * &y_t, want);
    }

    #[test]
    fn test_matrix_vector_multiply() {
        #[rustfmt::skip]
        let a = Matrix::from([
            [1.0, -1.0, 2.0],
            [0.0, -3.0, 1.0],
        ]);
        #[rustfmt::skip]
        let a_t = Matrix::from([
            [1.0, 0.0],
            [-1.0, -3.0],
            [2.0, 1.0],
        ]).transpose();
        let x = Vector::from([2.0, 1.0, 0.0]);

        let want = Vector::from([1.0, -3.0]);

        assert_eq!(&a * &x, want);
        assert_eq!(&a_t * &x, want);
    }

    #[test]
    fn test_transpose() {
        let x = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let y = Matrix::from([[1, 4], [2, 5], [3, 6]]);

        assert_eq!(x.transpose(), y);
    }

    #[test]
    fn test_reduce_dim() {
        let m = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let m2: Matrix<_, 1, 3> = m.view().reduce_dim::<0>(|x, y| x + y);
        assert_eq!(m2, Matrix::from([[5, 7, 9]]));
        let m3: Matrix<_, 2, 1> = m.view().reduce_dim::<1>(|x, y| x + y);
        assert_eq!(m3, Matrix::from([[6], [15]]));
    }
}
