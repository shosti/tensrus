use std::ops::Index;

use crate::{
    generic_tensor::GenericTensor,
    numeric::Numeric,
    shape::{reduced_shape, Shape},
    storage::{storage_idx, Layout},
    tensor::Tensor,
};

pub struct TensorView<'a, T: Numeric, const R: usize, const S: Shape> {
    pub(crate) storage: &'a [T],
    pub layout: Layout,
}

impl<'a, T: Numeric, const R: usize, const S: Shape> TensorView<'a, T, R, S> {
    pub fn reduce_dim<const DIM: usize>(
        self,
        f: impl Fn(T, T) -> T + 'static,
    ) -> GenericTensor<T, R, { reduced_shape(R, S, DIM) }> {
        GenericTensor::from_fn(|idx| {
            let mut src_idx = *idx;
            debug_assert!(src_idx[DIM] == 0);

            let mut res = self[&src_idx];
            for i in 1..S[DIM] {
                src_idx[DIM] = i;
                res = f(res, self[&src_idx]);
            }
            res
        })
    }
}

impl<'a, T: Numeric, const R: usize, const S: Shape> Index<&[usize; R]>
    for TensorView<'a, T, R, S>
{
    type Output = T;

    fn index(&self, idx: &[usize; R]) -> &Self::Output {
        let i = storage_idx(idx, S, self.layout).expect("out of bounds");
        self.storage.index(i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_dim() {
        let t = GenericTensor::<f64, 2, { [2, 3, 0, 0, 0, 0] }>::from([1, 2, 3, 4, 5, 6]);
        let t2 = t.view().reduce_dim::<0>(|x, y| x + y);
        assert_eq!(
            t2,
            GenericTensor::<f64, 2, { [1, 3, 0, 0, 0, 0] }>::from([5, 7, 9])
        );
        let t3 = t.view().reduce_dim::<1>(|x, y| x + y);
        assert_eq!(
            t3,
            GenericTensor::<f64, 2, { [2, 1, 0, 0, 0, 0] }>::from([6, 15])
        );
    }
}
