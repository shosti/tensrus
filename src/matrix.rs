use crate::{
    generic_tensor::GenericTensor,
    numeric::Numeric,
    shape::{Shape, Shaped},
    storage::{num_elems, storage_idx, IndexError, Layout, Storage, TensorStorage},
    tensor::{Tensor, TensorLike},
    type_assert::{Assert, IsTrue},
    vector::Vector,
    view::View,
};
use num::ToPrimitive;
use std::ops::{Index, Mul};

pub const fn matrix_shape(m: usize, n: usize) -> Shape {
    [m, n, 0, 0, 0, 0]
}

#[derive(Tensor, Clone)]
#[tensor_rank = 2]
#[tensor_shape = "matrix_shape(M, N)"]
pub struct Matrix<T: Numeric, const M: usize, const N: usize> {
    pub(crate) storage: Storage<T>,
    pub layout: Layout,
}

impl<T: Numeric, const M: usize, const N: usize> Matrix<T, M, N> {
    pub fn row(&self, i: usize) -> Result<Vector<T, N>, IndexError> {
        if i >= M {
            return Err(IndexError {});
        }

        Ok(Vector::from_fn(|[j]| self[&[i, *j]]))
    }

    pub fn col(&self, j: usize) -> Result<Vector<T, M>, IndexError> {
        if j >= N {
            return Err(IndexError {});
        }

        Ok(Vector::from_fn(|[i]| self[&[*i, j]]))
    }

    // This isn't terribly efficient, maybe think of a better way at some point?
    pub fn map_rows(mut self, f: impl Fn(Vector<T, N>) -> Vector<T, N>) -> Self {
        for i in 0..M {
            let v = f(Vector::from_fn(|[j]| self[&[i, *j]]));
            for j in 0..N {
                let idx = storage_idx(&[i, j], Self::shape(), self.layout).unwrap();
                self.storage[idx] = v[&[j]];
            }
        }
        self
    }

    pub fn normalize_rows(self) -> Self {
        self.map_rows(|v| v.normalize().into())
    }

    pub fn matrix_view(&self) -> MatrixView<T, M, N> {
        MatrixView {
            storage: &self.storage,
            layout: self.layout,
        }
    }

    pub fn transpose(self) -> Matrix<T, N, M> {
        Matrix {
            storage: self.storage,
            layout: self.layout.transpose(),
        }
    }

    /// Multiplies self * other and adds the result to out, returning out
    pub fn matmul_into<const P: usize>(
        &self,
        other: &Matrix<T, N, P>,
        out: Matrix<T, M, P>,
    ) -> Matrix<T, M, P> {
        matmul_with_initial_impl::<T, M, N, P>(
            &self.storage,
            self.layout,
            &other.storage,
            other.layout,
            out,
        )
    }

    /// Multiplies self * other and adds the result to out, returning out
    pub fn matmul_view_into<const P: usize>(
        &self,
        other: MatrixView<T, N, P>,
        out: Matrix<T, M, P>,
    ) -> Matrix<T, M, P> {
        matmul_with_initial_impl::<T, M, N, P>(
            &self.storage,
            self.layout,
            other.storage,
            other.layout,
            out,
        )
    }

    /// Multiplies self * x and adds the result to out, returning out
    pub fn matvecmul_into(&self, x: &Vector<T, N>, out: Vector<T, M>) -> Vector<T, M> {
        matvecmul_with_initial_impl::<T, M, N>(&self.storage, self.layout, &x.storage, out)
    }
}

impl<T: Numeric, const N: usize> Matrix<T, N, N> {
    pub fn identity() -> Self {
        Self::from_fn(|[i, j]| if i == j { T::one() } else { T::zero() })
    }
}

impl<T: Numeric, const M: usize, const N: usize, U> From<[[U; N]; M]> for Matrix<T, M, N>
where
    U: ToPrimitive,
{
    fn from(arrs: [[U; N]; M]) -> Self {
        arrs.into_iter().flatten().collect()
    }
}

impl<'a, T: Numeric, const M: usize, const N: usize> From<MatrixView<'a, T, M, N>>
    for Matrix<T, M, N>
{
    fn from(src: MatrixView<'a, T, M, N>) -> Self {
        Self {
            storage: src.storage.into(),
            layout: src.layout,
        }
    }
}

impl<T: Numeric, const M: usize, const N: usize, F: ToPrimitive> From<[F; M * N]>
    for Matrix<T, M, N>
{
    fn from(arr: [F; M * N]) -> Self {
        arr.into_iter().collect()
    }
}

impl<'a, T: Numeric, const M: usize, const N: usize, const P: usize> Mul<&'a Matrix<T, N, P>>
    for &'a Matrix<T, M, N>
{
    type Output = Matrix<T, M, P>;

    fn mul(self, other: &Matrix<T, N, P>) -> Self::Output {
        matmul_impl::<T, M, N, P>(&self.storage, self.layout, &other.storage, other.layout)
    }
}

#[derive(Debug)]
pub struct MatrixView<'a, T: Numeric, const M: usize, const N: usize> {
    pub(crate) storage: &'a [T],
    pub layout: Layout,
}

impl<'a, T: Numeric, const M: usize, const N: usize> TensorLike for MatrixView<'a, T, M, N> {
    type T = T;
}

impl<'a, T: Numeric, const M: usize, const N: usize> MatrixView<'a, T, M, N> {
    pub fn transpose(&self) -> MatrixView<'a, T, N, M> {
        MatrixView {
            storage: self.storage,
            layout: self.layout.transpose(),
        }
    }

    pub fn reshape<const M2: usize, const N2: usize>(&self) -> MatrixView<T, M2, N2>
    where
        Assert<{ num_elems(2, matrix_shape(M, N)) == num_elems(2, matrix_shape(M2, N2)) }>: IsTrue,
    {
        MatrixView {
            storage: self.storage,
            layout: self.layout,
        }
    }

    pub fn view(&self) -> View<'a, Matrix<T, M, N>> {
        View::new(self.storage, self.layout)
    }

    pub fn matmul_into<const P: usize>(
        self,
        other: &Matrix<T, N, P>,
        out: Matrix<T, M, P>,
    ) -> Matrix<T, M, P> {
        matmul_with_initial_impl::<T, M, N, P>(
            self.storage,
            self.layout,
            &other.storage,
            other.layout,
            out,
        )
    }

    pub fn matmul_view_into<const P: usize>(
        self,
        other: MatrixView<T, N, P>,
        out: Matrix<T, M, P>,
    ) -> Matrix<T, M, P> {
        matmul_with_initial_impl::<T, M, N, P>(
            self.storage,
            self.layout,
            other.storage,
            other.layout,
            out,
        )
    }

    /// Multiplies self * x and adds the result to out, returning out
    pub fn matvecmul_into(&self, x: &Vector<T, N>, out: Vector<T, M>) -> Vector<T, M> {
        matvecmul_with_initial_impl::<T, M, N>(self.storage, self.layout, &x.storage, out)
    }
}

impl<'a, T: Numeric, const M: usize, const N: usize> TensorStorage<T> for MatrixView<'a, T, M, N> {
    fn storage(&self) -> &[T] {
        self.storage
    }

    fn layout(&self) -> Layout {
        self.layout
    }
}

impl<'a, T: Numeric, const M: usize, const N: usize> Index<&[usize; 2]>
    for MatrixView<'a, T, M, N>
{
    type Output = T;

    fn index(&self, idx: &[usize; 2]) -> &Self::Output {
        let i = crate::storage::storage_idx(idx, matrix_shape(M, N), self.layout)
            .expect("out of bounds");
        self.storage.index(i)
    }
}

impl<'a, T: Numeric, const M: usize, const N: usize> Shaped for MatrixView<'a, T, M, N> {
    const R: usize = 2;
    const S: Shape = { matrix_shape(M, N) };
}

impl<'a, T: Numeric, const M: usize, const N: usize> Clone for MatrixView<'a, T, M, N> {
    fn clone(&self) -> Self {
        MatrixView {
            storage: self.storage,
            layout: self.layout,
        }
    }
}

fn matmul_impl<T: Numeric, const M: usize, const N: usize, const P: usize>(
    a_storage: &[T],
    a_transpose: Layout,
    b_storage: &[T],
    b_transpose: Layout,
) -> Matrix<T, M, P> {
    matmul_with_initial_impl::<T, M, N, P>(
        a_storage,
        a_transpose,
        b_storage,
        b_transpose,
        Matrix::zeros(),
    )
}

fn matmul_with_initial_impl<T: Numeric, const M: usize, const N: usize, const P: usize>(
    a_storage: &[T],
    a_transpose: Layout,
    b_storage: &[T],
    b_transpose: Layout,
    mut out: Matrix<T, M, P>,
) -> Matrix<T, M, P> {
    // We need the out format to be column-major; if it isn't, take (B^T * A*T)^T
    if !out.layout.is_transposed() {
        let out_t = matmul_with_initial_impl::<T, P, N, M>(
            b_storage,
            b_transpose.transpose(),
            a_storage,
            a_transpose.transpose(),
            out.transpose(),
        );
        return out_t.transpose();
    }

    debug_assert!(out.layout.is_transposed());

    unsafe {
        T::gemm(
            a_transpose.to_blas(),
            b_transpose.to_blas(),
            M as i32,
            P as i32,
            N as i32,
            T::one(),
            a_storage,
            if a_transpose.is_transposed() { M } else { N } as i32,
            b_storage,
            if b_transpose.is_transposed() { N } else { P } as i32,
            T::one(),
            &mut out.storage,
            M as i32,
        )
    }

    out
}

fn matvecmul_impl<T: Numeric, const M: usize, const N: usize>(
    a_storage: &[T],
    a_transpose: Layout,
    x_storage: &[T],
) -> Vector<T, M> {
    matvecmul_with_initial_impl::<T, M, N>(a_storage, a_transpose, x_storage, Vector::zeros())
}

fn matvecmul_with_initial_impl<T: Numeric, const M: usize, const N: usize>(
    a_storage: &[T],
    a_transpose: Layout,
    x_storage: &[T],
    mut out: Vector<T, M>,
) -> Vector<T, M> {
    debug_assert!(!out.layout.is_transposed());

    // BLAS always uses column-major format, so if we're "transposed" we're
    // already in BLAS format, otherwise we have to transpose.
    let trans = a_transpose.to_blas();
    let m = if a_transpose.is_transposed() { M } else { N } as i32;
    let n = if a_transpose.is_transposed() { N } else { M } as i32;
    let lda = m;

    unsafe {
        T::gemv(
            trans,
            m,
            n,
            T::one(),
            a_storage,
            lda,
            x_storage,
            1,
            T::one(),
            &mut out.storage,
            1,
        );
    }

    out
}

impl<'a, T: Numeric, const M: usize, const N: usize> Mul<&'a Vector<T, N>> for &'a Matrix<T, M, N> {
    type Output = Vector<T, M>;

    fn mul(self, other: &Vector<T, N>) -> Self::Output {
        matvecmul_impl::<T, M, N>(&self.storage, self.layout, &other.storage)
    }
}

impl<'a, T: Numeric, const M: usize, const N: usize> Mul<&'a Vector<T, N>>
    for MatrixView<'a, T, M, N>
{
    type Output = Vector<T, M>;

    fn mul(self, other: &Vector<T, N>) -> Self::Output {
        matvecmul_impl::<T, M, N>(self.storage, self.layout, &other.storage)
    }
}

impl<'a, T: Numeric, const M: usize, const N: usize, const P: usize> Mul<MatrixView<'a, T, N, P>>
    for &'a Matrix<T, M, N>
{
    type Output = Matrix<T, M, P>;

    fn mul(self, other: MatrixView<'a, T, N, P>) -> Self::Output {
        matmul_impl::<T, M, N, P>(&self.storage, self.layout, other.storage, other.layout)
    }
}

impl<'a, T: Numeric, const M: usize, const N: usize, const P: usize> Mul<&'a Matrix<T, N, P>>
    for MatrixView<'a, T, M, N>
{
    type Output = Matrix<T, M, P>;

    fn mul(self, other: &'a Matrix<T, N, P>) -> Self::Output {
        matmul_impl::<T, M, N, P>(self.storage, self.layout, &other.storage, other.layout)
    }
}

impl<'a, T: Numeric, const M: usize, const N: usize, const P: usize> Mul<MatrixView<'a, T, N, P>>
    for MatrixView<'a, T, M, N>
{
    type Output = Matrix<T, M, P>;

    fn mul(self, other: MatrixView<'a, T, N, P>) -> Self::Output {
        matmul_impl::<T, M, N, P>(self.storage, self.layout, other.storage, other.layout)
    }
}

impl<T: Numeric, const M: usize, const N: usize> From<GenericTensor<T, 2, { matrix_shape(M, N) }>>
    for Matrix<T, M, N>
{
    fn from(t: GenericTensor<T, 2, { matrix_shape(M, N) }>) -> Self {
        Self {
            storage: t.storage,
            layout: t.layout,
        }
    }
}

impl<T: Numeric, const M: usize, const N: usize> From<Matrix<T, M, N>>
    for GenericTensor<T, 2, { matrix_shape(M, N) }>
{
    fn from(t: Matrix<T, M, N>) -> Self {
        Self {
            storage: t.storage,
            layout: t.layout,
        }
    }
}

impl<T: Numeric, const M: usize, const N: usize> std::fmt::Debug for Matrix<T, M, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        writeln!(f, "Matrix<{}, {}> [", M, N)?;
        for i in 0..M {
            write!(f, "\t")?;
            for j in 0..N {
                write!(f, "{},\t", self[&[i, j]])?;
            }
            writeln!(f)?;
        }
        writeln!(f, "]")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::Vector;
    use proptest::prelude::*;
    use seq_macro::seq;

    #[test]
    #[rustfmt::skip]
    fn test_matrix_basics() {
        let x: Matrix<f64, _, _> = Matrix::from([
            [3, 4, 5],
            [2, 7, 9],
            [6, 5, 10],
            [3, 7, 3],
        ]);

        assert_eq!(x[&[2, 1]], 5.0);
        assert_eq!(x[&[3, 2]], 3.0);

        let y: Matrix<f64, 4, 3> = Matrix::from([
            3.0, 4.0, 5.0,
            2.0, 7.0, 9.0,
            6.0, 5.0, 10.0,
            3.0, 7.0, 3.0
        ]);
        assert_eq!(x, y);
    }

    #[test]
    fn test_add() {
        let x: Matrix<f64, _, _> = Matrix::from([[1, 2], [3, 4]]);
        let y = GenericTensor::<f64, 2, { matrix_shape(2, 2) }>::from([1, 2, 3, 4]);
        assert_eq!(x + y.view(), Matrix::from([[2, 4], [6, 8]]));
    }

    #[test]
    #[rustfmt::skip]
    fn test_from_iter() {
        let x: Matrix<f64, 3, 2> = [1, 2, 3].into_iter().cycle().collect();
        let y: Matrix<f64, _, _> = Matrix::from([
            [1.0, 2.0],
            [3.0, 1.0],
            [2.0, 3.0],
        ]);
        assert_eq!(x, y);
    }

    #[test]
    fn test_matmul_distributive() {
        let a: Matrix<f64, 1, 1> = Matrix::zeros();
        let b: Matrix<f64, 1, 2> = Matrix::zeros();
        let c: Matrix<f64, 1, 2> = Matrix::zeros();

        assert_eq!(&a * &(b.clone() + &c), (&a * &b) + &(&a * &c));
    }

    seq!(N in 1..=20 {
        proptest! {
            #[test]
            #[cfg(feature = "proptest")]
            fn test_identity_~N(v in prop::collection::vec(any::<f64>(), N)) {
                let i = Matrix::<f64, N, N>::identity();
                let x: Matrix::<f64, N, N> = v.into_iter().collect();

                assert_eq!(&i * &x, x);
                assert_eq!(&x * &i, x);
            }
        }
    });

    #[cfg(feature = "proptest")]
    fn assert_eq_within_tolerance<const M: usize, const N: usize>(
        a: Matrix<f64, M, N>,
        b: Matrix<f64, M, N>,
    ) {
        const TOLERANCE: f64 = 0.00001;
        for i in 0..M {
            for j in 0..N {
                assert!((a[&[i, j]] - b[&[i, j]]).abs() < TOLERANCE);
            }
        }
    }

    seq!(M in 1..=5 {
        seq!(N in 1..=5 {
            seq!(P in 1..=5 {
                proptest! {
                    #[test]
                    #[cfg(feature = "proptest")]
                    #[allow(clippy::identity_op)]
                    fn test_matmul_~M~N~P(v_a in proptest::collection::vec(-10000.0..10000.0, M * N),
                                          v_b in proptest::collection::vec(-10000.0..10000.0, N * P)) {
                        let a: Matrix::<f64, M, N> = v_a.into_iter().collect();
                        let b: Matrix::<f64, N, P> = v_b.into_iter().collect();

                        let c = &a * &b;

                        for i in 0..M {
                            for j in 0..P {
                                const TOLERANCE: f64 = 0.00001;
                                assert!((a.row(i).unwrap().dot(&b.col(j).unwrap()) - c[&[i, j]]).abs() < TOLERANCE);
                            }
                        }
                    }

                    #[test]
                    #[cfg(feature = "proptest")]
                    #[allow(clippy::identity_op)]
                    fn test_matmul_distributivity_~M~N~P(v_a in proptest::collection::vec(-10000.0..10000.0, M * N),
                                                         v_b in proptest::collection::vec(-10000.0..10000.0, N * P),
                                                         v_c in proptest::collection::vec(-10000.0..10000.0, N * P)) {
                        let a: Matrix::<f64, M, N> = v_a.into_iter().collect();
                        let b: Matrix::<f64, N, P> = v_b.into_iter().collect();
                        let c: Matrix::<f64, N, P> = v_c.into_iter().collect();

                        assert_eq_within_tolerance(&a * &(b.clone() + &c), (&a * &b) + &(&a * &c));
                    }

                    #[test]
                    #[cfg(feature = "proptest")]
                    #[allow(clippy::identity_op)]
                    fn test_matmul_transpose_~M~N~P(v_a in proptest::collection::vec(-10000.0..10000.0, M * N),
                                                    v_b in proptest::collection::vec(-10000.0..10000.0, N * P)) {
                        let a: Matrix::<f64, M, N> = v_a.into_iter().collect();
                        let b: Matrix::<f64, N, P> = v_b.into_iter().collect();

                        assert_eq_within_tolerance(&a * &b, (&b.clone().transpose() * &a.clone().transpose()).transpose());
                        assert_eq_within_tolerance(&a * b.matrix_view(), (&b.clone().transpose() * a.matrix_view().transpose()).transpose());
                        assert_eq_within_tolerance(a.matrix_view() * &b, (&b.clone().transpose() * a.matrix_view().transpose()).transpose());
                        assert_eq_within_tolerance(a.matrix_view() * b.matrix_view(), (b.matrix_view().transpose() * a.matrix_view().transpose()).transpose());
                    }
                }
            });
        });
    });

    seq!(N in 1..=20 {
        proptest! {
            #[test]
            #[cfg(feature = "proptest")]
            fn test_matvecmul_identity_~N(v in prop::collection::vec(any::<f64>(), N)) {
                let i = Matrix::<f64, N, N>::identity();
                let x: Vector::<f64, N> = v.into_iter().collect();

                assert_eq!(&i * &x, x);
            }
        }
    });

    seq!(M in 1..=10 {
        seq!(N in 1..=10 {
            proptest! {
                #[test]
                #[cfg(feature = "proptest")]
                #[allow(clippy::identity_op)]
                fn test_matvecmul_~M~N(v_a in proptest::collection::vec(-10000.0..10000.0, M * N),
                                       v_x in proptest::collection::vec(-10000.0..10000.0, N)) {
                    let a: Matrix::<f64, M, N> = v_a.into_iter().collect();
                    let x: Vector::<f64, N> = v_x.into_iter().collect();

                    let b = &a * &x;

                    for i in 0..M {
                        const TOLERANCE: f64 = 0.00001;
                        assert!((a.row(i).unwrap().dot(&x) - b[&[i]]).abs() < TOLERANCE);
                    }
                }
            }
        });
    });

    #[test]
    #[allow(clippy::zero_prefixed_literal)]
    fn test_from_fn() {
        let x: Matrix<f64, 3, 4> = Matrix::from_fn(|idx| {
            let [i, j] = idx;
            let s = format!("{}{}", i, j);
            s.parse().unwrap()
        });
        let y = Matrix::from([[00, 01, 02, 03], [10, 11, 12, 13], [20, 21, 22, 23]]);

        assert_eq!(x, y);
    }

    #[test]
    fn test_matrix_vector_conversions() {
        let x: Matrix<f64, _, _> = Matrix::from([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);

        assert_eq!(x.col(0), Ok(Vector::from([1, 4, 7, 10])));
        assert_eq!(x.col(1), Ok(Vector::from([2, 5, 8, 11])));
        assert_eq!(x.col(2), Ok(Vector::from([3, 6, 9, 12])));
        assert_eq!(x.col(3), Err(IndexError {}));
        assert_eq!(x.row(0), Ok(Vector::from([1, 2, 3])));
        assert_eq!(x.row(1), Ok(Vector::from([4, 5, 6])));
        assert_eq!(x.row(2), Ok(Vector::from([7, 8, 9])));
        assert_eq!(x.row(3), Ok(Vector::from([10, 11, 12])));
        assert_eq!(x.row(4), Err(IndexError {}));
    }

    #[test]
    fn test_matrix_multiply() {
        #[rustfmt::skip]
        let x: Matrix<f64, _, _> = Matrix::from([
            [1, 2],
            [3, 4],
            [5, 6]
        ]);
        #[rustfmt::skip]
        let x_t: Matrix<f64, _, _> = Matrix::from([
            [1, 3, 5],
            [2, 4, 6],
        ]).transpose();

        #[rustfmt::skip]
        let y: Matrix<f64, _, _> = Matrix::from([
            [7, 8, 9, 10],
            [9, 10, 11, 12]
        ]);
        #[rustfmt::skip]
        let y_t: Matrix<f64, _, _> = Matrix::from([
            [7, 9],
            [8, 10],
            [9, 11],
            [10, 12],
        ]).transpose();

        #[rustfmt::skip]
        let want: Matrix<f64, _, _> = Matrix::from([
            [25, 28, 31, 34],
            [57, 64, 71, 78],
            [89, 100, 111, 122]
        ]);

        // assert_eq!(&x * &y, want);
        assert_eq!(&x_t * &y, want);
        assert_eq!(&x * &y_t, want);
        assert_eq!(&x_t * &y_t, want);
    }

    #[test]
    fn test_matrix_vector_multiply() {
        #[rustfmt::skip]
        let a: Matrix<f64, _, _> = Matrix::from([
            [1, -1, 2],
            [0, -3, 1],
        ]);
        #[rustfmt::skip]
        let a_t: Matrix<f64, _, _> = Matrix::from([
            [1, 0],
            [-1, -3],
            [2, 1],
        ]).transpose();
        let x: Vector<f64, _> = Vector::from([2, 1, 0]);

        let want = Vector::from([1, -3]);

        assert_eq!(&a * &x, want);
        assert_eq!(&a_t * &x, want);
    }

    #[test]
    fn test_transpose() {
        let x: Matrix<f64, _, _> = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let y: Matrix<f64, _, _> = Matrix::from([[1, 4], [2, 5], [3, 6]]);

        assert_eq!(x.transpose(), y);
    }

    #[test]
    fn test_reduce_dim() {
        let m = Matrix::<f64, _, _>::from([[1, 2, 3], [4, 5, 6]]);
        let m2: Matrix<f64, 1, 3> = m.view().to_generic().reduce_dim::<0>(|x, y| x + y).into();
        assert_eq!(m2, Matrix::<f64, _, _>::from([[5, 7, 9]]));
        let m3: Matrix<f64, 2, 1> = m.view().to_generic().reduce_dim::<1>(|x, y| x + y).into();
        assert_eq!(m3, Matrix::<f64, _, _>::from([[6], [15]]));
    }
}
