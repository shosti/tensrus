use std::ops::Index;

use crate::{
    broadcast::{broadcast_compat, broadcast_normalize},
    generic_tensor::GenericTensor,
    numeric::Numeric,
    shape::{reduced_shape, Shape},
    storage::{storage_idx, Layout},
    tensor::Tensor,
    type_assert::{Assert, IsTrue},
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

    pub fn broadcast<const R_DEST: usize, const S_DEST: Shape>(
        self,
    ) -> GenericTensor<T, R_DEST, S_DEST>
    where
        Assert<{ broadcast_compat(R, S, R_DEST, S_DEST) }>: IsTrue,
    {
        let s_normalized = broadcast_normalize(S, R, R_DEST);
        GenericTensor::from_fn(|idx| {
            let mut src_idx = [0; R];
            let mut dim = 0;
            for i in 0..R_DEST {
                if s_normalized[i] == 1 && S_DEST[i] != 1 {
                    continue;
                }
                src_idx[dim] = idx[i];
                dim += 1;
            }

            self[&src_idx]
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
    use crate::{matrix::Matrix, shape::MAX_DIMS, vector::Vector};

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

    #[test]
    fn test_broadcast() {
        let v = Vector::<f64, _>::from([1, 2, 3]);
        let m: Matrix<_, 3, 3> = TensorView::from(&v).broadcast().into();
        assert_eq!(
            m,
            Matrix::<f64, _, _>::from([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        );

        let t: GenericTensor<_, 3, { [3; MAX_DIMS] }> = TensorView::from(&v).broadcast();
        assert_eq!(
            t,
            [1, 2, 3]
                .into_iter()
                .cycle()
                .collect::<GenericTensor<f64, 3, { [3; MAX_DIMS] }>>()
        );
    }
}
