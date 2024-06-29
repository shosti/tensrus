use std::ops::Index;

use crate::{
    errors::IndexError,
    shape::{self, Shape, MAX_DIMS},
    type_assert::{Assert, IsTrue},
};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    #[default]
    Normal,
    Transposed,
}

#[derive(Debug, Clone)]
pub struct Storage<T> {
    pub data: Box<[T]>,
    pub layout: Layout,
}

pub trait TensorStorage<T> {
    fn storage(&self) -> &Storage<T>;
}

pub trait TensorLayout {
    fn layout(&self) -> Layout;
}

pub trait OwnedTensorStorage<T>: TensorStorage<T> {
    fn into_storage(self) -> Storage<T>;
    fn from_storage(storage: Storage<T>) -> Self;
}

impl Layout {
    pub fn transpose(&self) -> Self {
        match self {
            Self::Normal => Self::Transposed,
            Self::Transposed => Self::Normal,
        }
    }

    pub fn to_blas(&self) -> u8 {
        match self {
            Layout::Normal => b'T',
            Layout::Transposed => b'N',
        }
    }

    pub fn is_transposed(&self) -> bool {
        match self {
            Layout::Normal => false,
            Layout::Transposed => true,
        }
    }
}

impl<T> Storage<T> {
    pub(crate) fn from_fn(f: impl Fn(usize) -> T, len: usize) -> Self {
        let data = (0..len).map(f).collect();

        Self {
            data,
            layout: Layout::default(),
        }
    }

    pub(crate) fn from_iter<I>(iter: I, n: usize) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Default,
    {
        let vals: Vec<T> = iter
            .into_iter()
            .chain(std::iter::repeat_with(|| T::default()))
            .take(n)
            .collect();

        Self {
            data: vals.into(),
            layout: Layout::default(),
        }
    }

    pub(crate) fn index(&self, idx: &[usize], rank: usize, shape: Shape) -> Result<&T, IndexError> {
        let i = self.storage_idx(idx, rank, shape)?;
        Ok(self.data.index(i))
    }

    pub(crate) fn get_nth_idx(
        n: usize,
        idx: &mut [usize],
        rank: usize,
        shape: Shape,
        layout: Layout,
    ) -> Result<(), IndexError> {
        if n >= shape::num_elems(rank, shape) {
            return Err(IndexError::OutOfBounds);
        }

        if layout == Layout::Transposed {
            Self::get_nth_idx(
                n,
                idx,
                rank,
                shape::transpose(rank, shape),
                layout.transpose(),
            )?;
            for i in 0..(rank / 2) {
                idx.swap(i, rank - i - 1);
            }
            return Ok(());
        }

        let mut i = n;
        let stride = shape::stride(rank, shape);
        for dim in 0..rank {
            let s = stride[dim];
            let cur = i / s;
            idx[dim] = cur;
            i -= cur * s;
        }
        debug_assert_eq!(i, 0);

        Ok(())
    }

    pub(crate) fn storage_idx(
        &self,
        idx: &[usize],
        rank: usize,
        shape: Shape,
    ) -> Result<usize, IndexError> {
        storage_idx(idx, rank, shape, self.layout)
    }

    pub(crate) fn transpose(self) -> Self {
        Self {
            data: self.data,
            layout: self.layout.transpose(),
        }
    }

    pub(crate) unsafe fn transmute<U>(self) -> Storage<U>
    where
        Assert<{ std::mem::size_of::<T>() == std::mem::size_of::<U>() }>: IsTrue,
    {
        let data = self.data;
        let len = data.len();
        let ptr = Box::into_raw(data) as *mut T;

        let slice_u: &mut [U] = std::slice::from_raw_parts_mut(ptr as *mut U, len);
        let data_u = Box::from_raw(slice_u as *mut [U]);

        Storage {
            data: data_u,
            layout: self.layout,
        }
    }
}

impl<T, const N: usize> From<[T; N]> for Storage<T> {
    fn from(arr: [T; N]) -> Self {
        Self {
            data: Box::new(arr),
            layout: Layout::default(),
        }
    }
}

pub(crate) fn storage_idx(
    idx: &[usize],
    rank: usize,
    shape: Shape,
    layout: Layout,
) -> Result<usize, IndexError> {
    if rank == 0 {
        return Ok(0);
    }
    for (dim, &cur) in idx.iter().enumerate() {
        if cur >= shape[dim] {
            return Err(IndexError::OutOfBounds);
        }
    }

    match layout {
        Layout::Normal => Ok(calc_storage_idx(idx, rank, shape)),
        Layout::Transposed => {
            let orig_shape = shape::transpose(rank, shape);
            let mut orig_idx = [0; MAX_DIMS];
            orig_idx[..rank].copy_from_slice(idx);
            orig_idx[..rank].reverse();

            Ok(calc_storage_idx(&orig_idx, rank, orig_shape))
        }
    }
}

fn calc_storage_idx(idx: &[usize], rank: usize, shape: Shape) -> usize {
    let stride = shape::stride(rank, shape);
    let mut i = 0;
    for (dim, &cur) in idx.iter().enumerate() {
        i += stride[dim] * cur;
    }

    i
}

impl<T, Tn: TensorStorage<T>> TensorStorage<T> for &Tn {
    fn storage(&self) -> &Storage<T> {
        (*self).storage()
    }
}

impl<Tn: TensorLayout> TensorLayout for &Tn {
    fn layout(&self) -> Layout {
        (*self).layout()
    }
}
