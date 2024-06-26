use crate::shape::{transpose_shape, Shape, MAX_DIMS};

pub type Storage<T> = Box<[T]>;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    #[default]
    Normal,
    Transposed,
}

impl Layout {
    pub fn transpose(self) -> Layout {
        match self {
            Layout::Normal => Layout::Transposed,
            Layout::Transposed => Layout::Normal,
        }
    }

    pub fn is_transposed(self) -> bool {
        self == Layout::Transposed
    }

    pub fn to_blas(self) -> u8 {
        match self {
            Layout::Normal => b'T',
            Layout::Transposed => b'N',
        }
    }
}

pub trait TensorStorage<T> {
    fn storage(&self) -> &[T];
    fn layout(&self) -> Layout;
}

#[derive(Debug, PartialEq)]
pub struct IndexError {}

pub const fn num_elems(r: usize, s: Shape) -> usize {
    let mut dim = 0;
    let mut n = 1;

    while dim < r {
        n *= s[dim];
        dim += 1;
    }

    n
}

pub(crate) fn storage_idx(
    r: usize,
    idx: &[usize],
    shape: Shape,
    layout: Layout,
) -> Result<usize, IndexError> {
    if r == 0 {
        return Ok(0);
    }
    for (dim, &cur) in idx.iter().enumerate() {
        if cur >= shape[dim] {
            return Err(IndexError {});
        }
    }

    match layout {
        Layout::Normal => Ok(calc_storage_idx(idx, r, shape)),
        Layout::Transposed => {
            let orig_shape = transpose_shape(r, shape);
            let mut orig_idx = [0; MAX_DIMS];
            orig_idx[..r].copy_from_slice(idx);
            orig_idx[..r].reverse();

            Ok(calc_storage_idx(&orig_idx, r, orig_shape))
        }
    }
}

fn calc_storage_idx(idx: &[usize], rank: usize, shape: Shape) -> usize {
    let stride = crate::shape::stride(rank, shape);
    let mut i = 0;
    for (dim, &cur) in idx.iter().enumerate() {
        i += stride[dim] * cur;
    }

    i
}

pub(crate) fn nth_idx<const R: usize>(
    n: usize,
    shape: Shape,
    layout: Layout,
) -> Result<[usize; R], IndexError> {
    let mut idx = [0; R];
    get_nth_idx(n, &mut idx, R, shape, layout)?;
    Ok(idx)
}

/// Finds the nth index for the given tensor parameters and copies it into idx.
pub(crate) fn get_nth_idx(
    n: usize,
    idx: &mut [usize],
    r: usize,
    shape: Shape,
    layout: Layout,
) -> Result<(), IndexError> {
    if n >= num_elems(r, shape) {
        return Err(IndexError {});
    }

    if layout == Layout::Transposed {
        get_nth_idx(n, idx, r, transpose_shape(r, shape), layout.transpose())?;
        for i in 0..(r / 2) {
            let tmp = idx[i];
            idx[i] = idx[r - i - 1];
            idx[r - i - 1] = tmp;
            return Ok(());
        }
    }

    let mut i = n;
    let stride = crate::shape::stride(r, shape);
    for dim in 0..r {
        let s = stride[dim];
        let cur = i / s;
        idx[dim] = cur;
        i -= cur * s;
    }
    debug_assert_eq!(i, 0);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nth_idx() {
        let shape = [3, 2, 0, 0, 0, 0];

        assert_eq!(nth_idx::<2>(0, shape, Layout::Normal).unwrap(), [0, 0]);
        assert_eq!(nth_idx::<2>(1, shape, Layout::Normal).unwrap(), [0, 1]);
        assert_eq!(nth_idx::<2>(2, shape, Layout::Normal).unwrap(), [1, 0]);
        assert_eq!(nth_idx::<2>(3, shape, Layout::Normal).unwrap(), [1, 1]);
        assert_eq!(nth_idx::<2>(4, shape, Layout::Normal).unwrap(), [2, 0]);
        assert_eq!(nth_idx::<2>(5, shape, Layout::Normal).unwrap(), [2, 1]);

        assert_eq!(nth_idx::<2>(0, shape, Layout::Transposed).unwrap(), [0, 0]);
        assert_eq!(nth_idx::<2>(1, shape, Layout::Transposed).unwrap(), [1, 0]);
        assert_eq!(nth_idx::<2>(2, shape, Layout::Transposed).unwrap(), [2, 0]);
        assert_eq!(nth_idx::<2>(3, shape, Layout::Transposed).unwrap(), [0, 1]);
        assert_eq!(nth_idx::<2>(4, shape, Layout::Transposed).unwrap(), [1, 1]);
        assert_eq!(nth_idx::<2>(5, shape, Layout::Transposed).unwrap(), [2, 1]);
    }
}
