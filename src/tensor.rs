use crate::{
    generic_tensor::{GenericTensor, IntoGeneric},
    iterator::Iter,
    scalar::Scalar,
    shape::{Shape, Shaped},
    storage::{self, Layout, OwnedTensorStorage, Storage, TensorLayout},
    view::View,
};
use num::{One, Zero};
use std::{fmt::Debug, ops::IndexMut};
use std::{iter::Sum, ops::Index};

pub trait TensorIndex:
    AsRef<[usize]>
    + AsMut<[usize]>
    + Copy
    + Debug
    + Index<usize, Output = usize>
    + IndexMut<usize>
    + 'static
{
    fn from_slice(s: &[usize]) -> Self;
    fn default() -> Self;
    fn transpose(self) -> Self;
}

pub trait Indexable:
    for<'a> Index<&'a Self::Idx, Output = Self::T> + Sized + Shaped + TensorLayout
{
    type Idx: TensorIndex;
    type T;

    fn next_idx(&self, idx: &Self::Idx) -> Option<Self::Idx> {
        let i =
            storage::storage_idx(idx.as_ref(), Self::rank(), Self::shape(), self.layout()).ok()?;
        if i >= Self::num_elems() - 1 {
            return None;
        }

        let mut idx = Self::Idx::default();
        Storage::<Self::T>::get_nth_idx(
            i + 1,
            idx.as_mut(),
            Self::rank(),
            Self::shape(),
            self.layout(),
        )
        .ok();
        Some(idx)
    }
}

pub trait Tensor: Indexable + OwnedTensorStorage<Self::T> {
    fn from_fn(f: impl Fn(&Self::Idx) -> Self::T) -> Self {
        let storage = Storage::from_fn(
            |i| {
                let mut idx = Self::Idx::default();
                Storage::<Self::T>::get_nth_idx(
                    i,
                    idx.as_mut(),
                    Self::rank(),
                    Self::shape(),
                    Layout::default(),
                )
                .unwrap();
                f(&idx)
            },
            Self::num_elems(),
        );

        Self::from_storage(storage)
    }

    fn set(self, idx: &Self::Idx, f: impl Fn(&Self::T) -> Self::T) -> Self {
        let mut storage = self.into_storage();
        let i = storage
            .storage_idx(idx.as_ref(), Self::rank(), Self::shape())
            .unwrap();
        storage.data[i] = f(storage.data.index(i));

        Self::from_storage(storage)
    }

    fn map(mut self, f: impl Fn(&Self::Idx, &Self::T) -> Self::T) -> Self {
        let mut next_idx = Some(Self::Idx::default());
        while let Some(idx) = next_idx {
            self = self.set(&idx, |val| f(&idx, val));
            next_idx = self.next_idx(&idx);
        }

        self
    }

    fn iter(&self) -> Iter<Self> {
        Iter::new(self)
    }

    fn repeat(n: Self::T) -> Self
    where
        Self::T: Copy,
    {
        Self::from_fn(|_| n)
    }

    fn zeros() -> Self
    where
        Self::T: Zero + Copy,
    {
        Self::repeat(Self::T::zero())
    }

    fn ones() -> Self
    where
        Self::T: One + Copy,
    {
        Self::repeat(Self::T::one())
    }

    fn view(&self) -> View<Self> {
        View::new(self.storage())
    }

    fn as_generic<const R: usize, const S: Shape>(&self) -> View<GenericTensor<Self::T, R, S>>
    where
        Self: IntoGeneric<Self::T, R, S>,
    {
        View::new(self.storage())
    }

    fn relu(self) -> Self
    where
        Self::T: Copy + Zero + PartialOrd,
    {
        self.map(|_, val| {
            if *val < Self::T::zero() {
                Self::T::zero()
            } else {
                *val
            }
        })
    }

    fn sum(&self) -> Scalar<Self::T>
    where
        Self::T: Sum + Copy,
    {
        let s: Self::T = self.iter().values().copied().sum();
        Scalar::from(s)
    }
}

impl<const R: usize> TensorIndex for [usize; R] {
    fn from_slice(s: &[usize]) -> Self {
        let mut ret = [0; R];
        ret.copy_from_slice(&s[..R]);
        ret
    }

    fn default() -> Self {
        [0; R]
    }

    fn transpose(mut self) -> Self {
        self.reverse();
        self
    }
}

impl<Tn: Shaped> Shaped for &Tn {
    fn rank() -> usize {
        Tn::rank()
    }

    fn shape() -> Shape {
        Tn::shape()
    }
}

impl<'a, Tn> Indexable for &'a Tn
where
    Tn: Indexable,
    &'a Tn: for<'b> Index<&'b Tn::Idx, Output = Tn::T>,
{
    type Idx = Tn::Idx;
    type T = Tn::T;

    fn next_idx(&self, idx: &Self::Idx) -> Option<Self::Idx> {
        (*self).next_idx(idx)
    }
}
