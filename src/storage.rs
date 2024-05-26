use crate::{
    tensor::{transpose_shape, IndexError, Shape},
    tensor2::Layout,
};

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
    let stride = crate::tensor::stride(rank, shape);
    let mut i = 0;
    for (dim, &cur) in idx.iter().enumerate() {
        i += stride[dim] * cur;
    }

    i
}
