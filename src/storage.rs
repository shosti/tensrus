use crate::{
    shape::{transpose_shape, Shape},
};

pub type Storage<T> = Box<[T]>;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    #[default]
    Normal,
    Transposed,
}

#[derive(Debug, PartialEq)]
pub struct IndexError {}

pub(crate) const fn num_elems(r: usize, s: Shape) -> usize {
    let mut dim = 0;
    let mut n = 1;

    while dim < r {
        n *= s[dim];
        dim += 1;
    }

    n
}

pub(crate) fn storage_idx<const R: usize>(
    idx: &[usize; R],
    shape: Shape,
    layout: Layout,
) -> Result<usize, IndexError> {
    if R == 0 {
        return Ok(0);
    }
    for (dim, &cur) in idx.iter().enumerate() {
        if cur >= shape[dim] {
            return Err(IndexError {});
        }
    }

    match layout {
        Layout::Normal => Ok(calc_storage_idx(idx, R, shape)),
        Layout::Transposed => {
            let orig_shape = transpose_shape(R, shape);
            let mut orig_idx = *idx;
            orig_idx.reverse();

            Ok(calc_storage_idx(&orig_idx, R, orig_shape))
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
    if n >= num_elems(R, shape) {
        return Err(IndexError {});
    }

    if layout == Layout::Transposed {
        let mut t_idx = nth_idx(n, transpose_shape(R, shape), Layout::Normal).unwrap();
        t_idx.reverse();
        return Ok(t_idx);
    }

    let mut i = n;
    let stride = crate::shape::stride(R, shape);
    let mut res = [0; R];
    for dim in 0..R {
        let s = stride[dim];
        let cur = i / s;
        res[dim] = cur;
        i -= cur * s;
    }
    debug_assert_eq!(i, 0);

    Ok(res)
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
