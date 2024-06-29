use crate::{
    generic_tensor::{GenericTensor, IntoGeneric},
    iterator::Iter,
    matrix::{IntoMatrix, Matrix},
    shape::{subtensor_shape, Broadcastable, Reducible, Shape, Shaped, Transposable},
    storage::{self, Layout, Storage, TensorLayout, TensorStorage},
    tensor::{Indexable, Tensor, TensorIndex},
    translation::Translation,
};
use std::{convert::identity, ops::Add, ops::Index};

pub struct View<'a, Tn: Tensor> {
    storage: &'a Storage<Tn::T>,
    is_transposed: bool,
}

impl<'a, Tn: Tensor + 'a> View<'a, Tn> {
    pub(crate) fn new(storage: &'a Storage<Tn::T>) -> Self {
        Self {
            storage,
            is_transposed: false,
        }
    }

    pub fn translate<Dest>(self, f: impl Fn(Dest::Idx) -> Tn::Idx + 'a) -> Translation<'a, Dest>
    where
        Dest: Tensor<T = Tn::T>,
    {
        self.as_translation().translate(f)
    }

    pub fn as_translation(self) -> Translation<'a, Tn> {
        if !self.is_transposed {
            return Translation::<Tn>::new::<Tn>(self.storage, identity);
        }
        let orig_layout = self.storage.layout;
        let self_layout = self.layout();
        let t = move |idx: Tn::Idx| {
            let n =
                storage::storage_idx(idx.as_ref(), Tn::rank(), Tn::shape(), self_layout).unwrap();
            let mut out = Tn::Idx::default();
            Storage::<Tn::T>::get_nth_idx(n, out.as_mut(), Tn::rank(), Tn::shape(), orig_layout)
                .unwrap();

            out
        };

        Translation::<Tn>::new::<Tn>(self.storage, t)
    }

    pub fn iter(&self) -> Iter<Self> {
        Iter::new(self)
    }

    pub fn to_owned(&self) -> Tn
    where
        Tn::T: Clone,
    {
        Tn::from_fn(|idx| self[idx].clone())
    }

    fn storage_idx(&self, idx: &Tn::Idx) -> usize {
        Self::calc_storage_idx(idx, self.layout())
    }

    pub(crate) fn calc_storage_idx(idx: &Tn::Idx, layout: Layout) -> usize {
        storage::storage_idx(idx.as_ref(), Tn::rank(), Tn::shape(), layout).unwrap()
    }

    pub fn as_generic<const R: usize, const S: Shape>(self) -> View<'a, GenericTensor<Tn::T, R, S>>
    where
        Tn: IntoGeneric<Tn::T, R, S>,
    {
        View {
            storage: self.storage,
            is_transposed: self.is_transposed,
        }
    }

    pub fn as_matrix<const M: usize, const N: usize>(self) -> View<'a, Matrix<Tn::T, M, N>>
    where
        Tn: IntoMatrix<Tn::T, M, N>,
    {
        View {
            storage: self.storage,
            is_transposed: self.is_transposed,
        }
    }

    pub fn reduce_dim<Dest, const DIM: usize>(
        self,
        f: impl Fn(Tn::T, Tn::T) -> Tn::T + 'static,
    ) -> Dest
    where
        Tn: Reducible<Dest, DIM>,
        Tn::T: Copy,
        Dest: Tensor<T = Tn::T, Idx = Tn::Idx>,
    {
        self.as_translation().reduce_dim(f)
    }

    pub fn broadcast<Dest>(self) -> Translation<'a, Dest>
    where
        Dest: Tensor<T = Tn::T>,
        Tn: Broadcastable<Dest>,
    {
        self.as_translation().broadcast()
    }
}

impl<'a, Tn, Rhs> PartialEq<Rhs> for View<'a, Tn>
where
    Tn: Tensor,
    Rhs: Indexable<T = Tn::T, Idx = Tn::Idx>,
    Tn::T: PartialEq,
{
    fn eq(&self, other: &Rhs) -> bool {
        self.iter().all(|(idx, val)| val == other.index(&idx))
    }
}

impl<'a, Tn: Tensor> Eq for View<'a, Tn> where Tn::T: Eq {}

impl<'a, T, const R: usize, const S: Shape> View<'a, GenericTensor<T, R, S>> {
    pub fn from_generic<Tn>(self) -> View<'a, Tn>
    where
        Tn: Tensor<T = T> + IntoGeneric<T, R, S>,
    {
        View {
            storage: self.storage,
            is_transposed: self.is_transposed,
        }
    }

    pub fn subtensor<'b: 'a, const N: usize>(
        self,
        idx: &'b [usize; N],
    ) -> Translation<'a, GenericTensor<T, { R - N }, { subtensor_shape(N, R, S) }>> {
        self.as_translation().subtensor(idx)
    }
}

impl<'a, T, const M: usize, const N: usize> View<'a, Matrix<T, M, N>> {
    pub fn from_matrix<Tn>(self) -> View<'a, Tn>
    where
        Tn: Tensor<T = T> + IntoMatrix<T, M, N>,
    {
        View {
            storage: self.storage,
            is_transposed: self.is_transposed,
        }
    }
}

impl<'a, 'b, Tn> Index<&'b Tn::Idx> for View<'a, Tn>
where
    Tn: Tensor,
{
    type Output = Tn::T;

    fn index(&self, idx: &'b Tn::Idx) -> &Self::Output {
        self.storage.data.index(self.storage_idx(idx))
    }
}

impl<'a, Tn> Shaped for View<'a, Tn>
where
    Tn: Tensor,
{
    fn shape() -> Shape {
        Tn::shape()
    }

    fn rank() -> usize {
        Tn::rank()
    }
}

impl<'a, Tn> TensorLayout for View<'a, Tn>
where
    Tn: Tensor,
{
    fn layout(&self) -> Layout {
        if self.is_transposed {
            self.storage.layout.transpose()
        } else {
            self.storage.layout
        }
    }
}

impl<'a, Tn> TensorStorage<Tn::T> for View<'a, Tn>
where
    Tn: Tensor,
{
    fn storage(&self) -> &Storage<Tn::T> {
        self.storage
    }
}

impl<'a, Tn> Indexable for View<'a, Tn>
where
    Tn: Tensor,
{
    type Idx = Tn::Idx;
    type T = Tn::T;
}

impl<'a, Tn, Dest> Transposable<View<'a, Dest>> for View<'a, Tn>
where
    Tn: Tensor + Transposable<Dest>,
    Dest: Tensor<T = Tn::T>,
{
    fn transpose(self) -> View<'a, Dest> {
        View {
            storage: self.storage,
            is_transposed: !self.is_transposed,
        }
    }
}

impl<'a, Tn: Tensor> Clone for View<'a, Tn> {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage,
            is_transposed: self.is_transposed,
        }
    }
}

impl<'a, Tn: Tensor> std::fmt::Debug for View<'a, Tn>
where
    Tn::T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        f.debug_struct("View ")
            .field("storage", self.storage)
            .finish()
    }
}

impl<'a, Tn, Rhs> Add<Rhs> for View<'a, Tn>
where
    Tn: Tensor,
    Rhs: Indexable<T = Tn::T, Idx = Tn::Idx>,
    Tn::T: Clone + Add<Output = Tn::T>,
{
    type Output = Tn;

    fn add(self, rhs: Rhs) -> Tn {
        Tn::from_fn(|idx| self[idx].clone() + rhs[idx].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_transpose(v in prop::collection::vec(any::<f64>(), 7 * 12)) {
            let m: Matrix<_, 7, 12> = v.into_iter().collect();
            let m_t = m.clone().transpose();
            assert_eq!(m.view().transpose(), m_t);
            assert_eq!(m.view().transpose().as_translation(), m_t);
            assert_eq!(m.view().as_translation().transpose(), m_t);
            assert_eq!(m.view().transpose().as_translation().transpose(), m);
            assert_eq!(m.clone().transpose().view().as_translation(), m_t);
            assert_eq!(m.clone().transpose().view().as_translation().transpose(), m);
            assert_eq!(m.clone().transpose().view().as_translation().transpose().transpose(), m_t);
            assert_eq!(m.clone().transpose().view().transpose().as_translation().transpose(), m_t);
        }
    }
}
