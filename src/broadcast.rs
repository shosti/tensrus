use crate::{
    generic_tensor::GenericTensor,
    numeric::Numeric,
    shape::{self, reduced_shape, Shape, MAX_DIMS},
    storage::Layout,
    tensor::Tensor,
    tensor_view::TensorView,
    type_assert::{Assert, IsTrue},
};
use std::{fmt::Debug, marker::PhantomData, ops::Index};

pub const fn broadcast_compat(r_src: usize, s_src: Shape, r_dest: usize, s_dest: Shape) -> bool {
    assert!(r_dest >= r_src, "cannot broadcast to lower dimension");

    let s_src_n = broadcast_normalize(s_src, r_src, r_dest);

    let mut i = 0;
    while i < r_dest {
        if s_src_n[i] != 1 && s_src_n[i] != s_dest[i] {
            return false;
        }
        i += 1;
    }

    true
}

pub const fn broadcast_normalize(s: Shape, r_src: usize, r_dest: usize) -> Shape {
    assert!(r_dest >= r_src, "cannot broadcast to a lower dimension");
    assert!(
        r_src <= shape::MAX_DIMS && r_dest <= shape::MAX_DIMS,
        "cannot broadcast to a dimension higher than the max"
    );

    let r_diff = r_dest - r_src;
    let mut ret = [0; shape::MAX_DIMS];

    let mut i = 0;
    while i < r_diff {
        ret[i] = 1;
        i += 1;
    }
    while i < r_dest {
        ret[i] = s[i - r_diff];
        i += 1;
    }

    ret
}

pub struct Broadcast<'a, Tn: Tensor> {
    _marker: PhantomData<Tn>,
    pub(crate) storage: &'a [Tn::T],
    pub layout: Layout,
    r_src: usize,
    s_src: Shape,
}

impl<'a, Tn: Tensor> Broadcast<'a, Tn> {
    fn idx_translate(&self, idx: &Tn::Idx) -> Shape {
        let r_dest = Tn::rank();
        let s_dest = Tn::shape();
        let s_normalized = broadcast_normalize(self.s_src, self.r_src, r_dest);

        let mut src_idx = [0; MAX_DIMS];
        let mut dim = 0;
        for i in 0..r_dest {
            if s_normalized[i] == 1 && s_dest[i] != 1 {
                continue;
            }
            src_idx[dim] = idx.as_ref()[i];
            dim += 1;
        }

        src_idx
    }
}

impl<'a, Tn: Tensor> Index<Tn::Idx> for Broadcast<'a, Tn> {
    type Output = Tn::T;

    fn index(&self, idx: Tn::Idx) -> &Self::Output {
        let idx_t = self.idx_translate(&idx);
        let i =
            crate::storage::storage_idx(&idx_t, self.s_src, self.layout).expect("out of bounds");
        self.storage.index(i)
    }
}

pub trait Broadcastable<T: Numeric, const R: usize, const S: Shape>
where
    for<'a> TensorView<'a, T, R, S>: From<&'a Self>,
{
    fn broadcast<const R_DEST: usize, const S_DEST: Shape>(
        &self,
    ) -> Broadcast<GenericTensor<T, R_DEST, S_DEST>>
    where
        Assert<{ broadcast_compat(R, S, R_DEST, S_DEST) }>: IsTrue,
    {
        let view = TensorView::from(self);
        Broadcast {
            storage: view.storage,
            layout: view.layout,
            r_src: R,
            s_src: S,
            _marker: PhantomData,
        }
    }
}

pub trait Reducible<T: Numeric, const R: usize, const S: Shape>
where
    for<'a> TensorView<'a, T, R, S>: From<&'a Self>,
{
    fn reduce_dim<const DIM: usize>(
        &self,
        f: impl Fn(T, T) -> T + 'static,
    ) -> GenericTensor<T, R, { reduced_shape(R, S, DIM) }> {
        let view: TensorView<T, R, S> = self.into();
        view.reduce_dim(f)
    }

    fn dim_sum<const DIM: usize>(&self) -> GenericTensor<T, R, { reduced_shape(R, S, DIM) }> {
        self.reduce_dim(|x, y| x + y)
    }

    fn dim_mul<const DIM: usize>(&self) -> GenericTensor<T, R, { reduced_shape(R, S, DIM) }> {
        self.reduce_dim(|x, y| x * y)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_broadcast_normalize() {
        assert_eq!(
            broadcast_normalize([7, 2, 3, 0, 0, 0], 3, 3),
            [7, 2, 3, 0, 0, 0],
        );

        assert_eq!(
            broadcast_normalize([7, 2, 3, 0, 0, 0], 3, 6),
            [1, 1, 1, 7, 2, 3],
        );
    }

    #[test]
    fn test_broadcast_compat() {
        let s = [1000, 256, 256, 256, 0, 0];
        let r = 4;

        assert!(broadcast_compat(4, [1000, 256, 1, 256, 0, 0], r, s));
        assert!(broadcast_compat(4, [1000, 1, 256, 256, 0, 0], r, s));
        assert!(broadcast_compat(2, [256, 1, 0, 0, 0, 0], r, s));
        assert!(!broadcast_compat(3, [1000, 256, 256, 0, 0, 0], r, s));
        assert!(!broadcast_compat(r, s, 4, [1000, 256, 1, 256, 0, 0]));
    }
}
