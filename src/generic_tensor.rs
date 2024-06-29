use crate::differentiable::{Differentiable, DifferentiableTensor};
use crate::matrix::{IntoMatrix, Matrix};
use crate::shape::{
    self, broadcast_compat, reduced_shape, Broadcastable, Reducible, Shape, Shaped, Transposable,
};
use crate::storage::{OwnedTensorStorage, Storage};
use crate::tensor::{Indexable, Tensor};
use crate::type_assert::{Assert, IsTrue};
use std::ops::Index;

#[derive(TensorStorage, OwnedTensorStorage, Tensor, Debug, Clone)]
pub struct GenericTensor<T, const R: usize, const S: Shape> {
    storage: Storage<T>,
}

impl<T, const R: usize, const S: Shape> GenericTensor<T, R, S> {
    pub fn reshape<const R2: usize, const S2: Shape>(self) -> GenericTensor<T, R2, S2>
    where
        Assert<{ shape::num_elems(R, S) == shape::num_elems(R2, S2) }>: IsTrue,
    {
        GenericTensor {
            storage: self.storage,
        }
    }

    pub fn from_generic<Tn>(self) -> Tn
    where
        Tn: IntoGeneric<T, R, S>,
    {
        Tn::from_storage(self.storage)
    }

    /// Transmutes a GenericTensor<T> to a GenericTensor<U> by re-interpreting
    /// the raw bits (without copying the underlying data).
    ///
    /// # Safety
    ///
    /// The following properties are guaranteed by the type signature:
    ///
    /// - T and U are the same size
    /// - T and U are Copy, so in practice they will be "primitive-ish"
    ///
    /// For everything else, you're on your own. Since the raw bits of T will be
    /// reinterpreted, there are a myriad of ways that undefined behavior can
    /// happen. See [the docs for
    /// std::mem::transmute](https://doc.rust-lang.org/std/mem/fn.transmute.html)
    /// for some ideas.
    pub unsafe fn transmute<U>(self) -> GenericTensor<U, R, S>
    where
        T: Copy,
        U: Copy,
        Assert<{ std::mem::size_of::<T>() == std::mem::size_of::<U>() }>: IsTrue,
    {
        GenericTensor {
            storage: self.storage.transmute(),
        }
    }
}

impl<T, const R: usize, const S: Shape>
    Transposable<GenericTensor<T, R, { shape::transpose(R, S) }>> for GenericTensor<T, R, S>
where
    Assert<{ R >= 2 }>: IsTrue,
{
    fn transpose(self) -> GenericTensor<T, R, { shape::transpose(R, S) }> {
        GenericTensor {
            storage: self.storage.transpose(),
        }
    }
}

impl<T, const R: usize, const S: Shape> Shaped for GenericTensor<T, R, S> {
    fn rank() -> usize {
        R
    }

    fn shape() -> Shape {
        S
    }
}

pub trait IntoGeneric<T, const R: usize, const S: Shape>: OwnedTensorStorage<T> + Sized {
    fn into_generic(self) -> GenericTensor<T, R, S> {
        GenericTensor {
            storage: self.into_storage(),
        }
    }
}

impl<T, const R: usize, const S: Shape> IntoGeneric<T, R, S> for GenericTensor<T, R, S> {}

impl<T, const M: usize, const N: usize> IntoMatrix<T, M, N>
    for GenericTensor<T, 2, { Matrix::<T, M, N>::shape() }>
{
}

impl<T, const R: usize, const S: Shape> From<[T; shape::num_elems(R, S)]>
    for GenericTensor<T, R, S>
{
    fn from(vals: [T; shape::num_elems(R, S)]) -> Self {
        Self {
            storage: vals.into(),
        }
    }
}

impl<T, const R: usize, const S: Shape> Index<&[usize; R]> for GenericTensor<T, R, S> {
    type Output = T;

    fn index(&self, idx: &[usize; R]) -> &Self::Output {
        self.storage.index(idx, R, S).unwrap()
    }
}

impl<T, const R: usize, const S: Shape> Index<&[usize; R]> for &GenericTensor<T, R, S> {
    type Output = T;

    fn index(&self, idx: &[usize; R]) -> &Self::Output {
        (*self).index(idx)
    }
}

impl<T, const R: usize, const S: Shape> Indexable for GenericTensor<T, R, S> {
    type Idx = [usize; R];
    type T = T;
}

impl<T, const R: usize, const S: Shape> DifferentiableTensor for GenericTensor<T, R, S> where
    T: Differentiable
{
}

impl<T, const R: usize, const S: Shape, const DIM: usize>
    Reducible<GenericTensor<T, R, { reduced_shape(R, S, DIM) }>, DIM> for GenericTensor<T, R, S>
{
}

impl<T, const R: usize, const S: Shape, const R_DEST: usize, const S_DEST: Shape>
    Broadcastable<GenericTensor<T, R_DEST, S_DEST>> for GenericTensor<T, R, S>
where
    Assert<{ broadcast_compat(R, S, R_DEST, S_DEST) }>: IsTrue,
{
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_basics() {
        let mut t: GenericTensor<i32, 3, { shape::rank3([2, 3, 4]) }> = GenericTensor::default();
        t = t.set(&[1, 1, 1], |_| 7);
        assert_eq!(t[&[1, 1, 1]], 7);
        assert_eq!(t[&[1, 1, 2]], 0);
    }

    #[test]
    fn test_equal() {
        let a: GenericTensor<_, 0, { [0; shape::MAX_DIMS] }> = GenericTensor::from([1]);
        let b: GenericTensor<_, 0, { [0; shape::MAX_DIMS] }> = GenericTensor::from([2]);
        assert_ne!(a, b);

        let x: GenericTensor<_, 2, { [2; 6] }> = GenericTensor::from([1, 2, 3, 4]);
        let y: GenericTensor<_, 2, { [2; 6] }> = GenericTensor::from([1, 2, 3, 5]);
        assert_ne!(x, y);
    }

    #[test]
    fn test_from_iterator() {
        let xs: [i64; 3] = [1, 2, 3];
        let iter = xs.iter().cycle().copied();

        let t1: GenericTensor<_, 0, { [0; shape::MAX_DIMS] }> = iter.clone().collect();
        assert_eq!(
            t1,
            GenericTensor::<_, 0, { [0; shape::MAX_DIMS] }>::from([1])
        );

        let t2: GenericTensor<_, 2, { [4, 2, 0, 0, 0, 0] }> = iter.clone().collect();
        assert_eq!(
            t2,
            GenericTensor::<_, 2, { [4, 2, 0, 0, 0, 0] }>::from([1, 2, 3, 1, 2, 3, 1, 2])
        );

        let t3: GenericTensor<_, 2, { [4, 2, 0, 0, 0, 0] }> = xs.iter().copied().collect();
        assert_eq!(
            t3,
            GenericTensor::<_, 2, { [4, 2, 0, 0, 0, 0] }>::from([1, 2, 3, 0, 0, 0, 0, 0])
        );
    }

    #[test]
    #[allow(clippy::zero_prefixed_literal)]
    fn test_from_fn() {
        let f = |idx: &[usize; 3]| {
            let [i, j, k] = *idx;
            let s = format!("{}{}{}", i, j, k);
            s.parse().unwrap()
        };
        let t1: GenericTensor<_, 3, { [2; 6] }> = GenericTensor::from_fn(f);
        assert_eq!(
            t1,
            GenericTensor::<_, 3, { [2; 6] }>::from([000, 001, 010, 011, 100, 101, 110, 111]),
        );

        let t2: GenericTensor<_, 3, { [1, 2, 3, 0, 0, 0] }> = GenericTensor::from_fn(f);
        assert_eq!(
            t2,
            GenericTensor::<_, 3, { [1, 2, 3, 0, 0, 0] }>::from([000, 001, 002, 010, 011, 012]),
        );

        let t3: GenericTensor<_, 3, { [3, 2, 1, 0, 0, 0] }> = GenericTensor::from_fn(f);
        assert_eq!(
            t3,
            GenericTensor::<_, 3, { [3, 2, 1, 0, 0, 0] }>::from([000, 010, 100, 110, 200, 210]),
        );

        let t4: GenericTensor<_, 3, { [2, 3, 1, 0, 0, 0] }> = GenericTensor::from_fn(f);
        assert_eq!(
            t4,
            GenericTensor::<_, 3, { [2, 3, 1, 0, 0, 0] }>::from([000, 010, 020, 100, 110, 120]),
        );
    }

    #[test]
    fn test_math() {
        let mut x: GenericTensor<_, 3, { [1, 2, 2, 0, 0, 0] }> = GenericTensor::from([1, 2, 3, 4]);
        let y: GenericTensor<_, 3, { [1, 2, 2, 0, 0, 0] }> = GenericTensor::from([5, 6, 7, 8]);
        let a: GenericTensor<_, 3, { [1, 2, 2, 0, 0, 0] }> = GenericTensor::from([6, 8, 10, 12]);

        assert_eq!(x.clone() + &y, a);

        x = x + &y;
        assert_eq!(x.clone(), a);

        let b: GenericTensor<_, 3, { [1, 2, 2, 0, 0, 0] }> = GenericTensor::from([12, 16, 20, 24]);
        assert_eq!(x.clone() * 2, b);

        x = x * 2;
        assert_eq!(x, b);
    }

    #[test]
    fn test_to_iter() {
        let t: GenericTensor<_, 2, { [2; 6] }> = (0..4).collect();
        let vals: Vec<_> = t.iter().values().copied().collect();
        assert_eq!(vals, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_map() {
        let t: GenericTensor<_, 2, { [2; 6] }> = GenericTensor::from([1, 2, 3, 4]);
        let u = t.map(|_, val| val * 2);

        let want: GenericTensor<_, 2, { [2; 6] }> = GenericTensor::from([2, 4, 6, 8]);
        assert_eq!(u, want);
    }

    #[test]
    #[rustfmt::skip]
    fn test_transpose() {
        let t: GenericTensor<_, 2, { [3, 2, 0, 0, 0, 0] }> = GenericTensor::from([
            1, 2,
            3, 4,
            5, 6,
        ]);
        let t2 = t.transpose().map(|_, v| v + 1);
        let want: GenericTensor<_, 2, { [2, 3, 0, 0, 0, 0] }> = GenericTensor::from([
            2, 4, 6,
            3, 5, 7,
        ]);

        assert_eq!(t2, want);
    }

    #[test]
    fn test_reshape() {
        let t = GenericTensor::<_, 2, { [3, 2, 0, 0, 0, 0] }>::from([1, 2, 3, 4, 5, 6]);
        let t2 = t.clone().reshape::<2, { [2, 3, 0, 0, 0, 0] }>();
        let t3 = t.clone().reshape::<1, { [6, 0, 0, 0, 0, 0] }>();

        assert_eq!(t[&[1, 0]], t2[&[0, 2]]);
        assert_eq!(t[&[2, 1]], t3[&[5]]);
    }

    #[test]
    fn test_transmute() {
        let x =
            GenericTensor::<bool, 2, { shape::rank2([2, 2]) }>::from([false, true, true, false]);
        let y: GenericTensor<u8, 2, { shape::rank2([2, 2]) }> = unsafe { x.transmute() };

        assert_eq!(
            y,
            GenericTensor::<u8, 2, { shape::rank2([2, 2]) }>::from([0, 1, 1, 0]),
        );
    }
}
