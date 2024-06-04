use crate::{
    generic_tensor::GenericTensor,
    shape::reduced_shape,
    storage::Layout,
    tensor::{ShapedTensor, Tensor},
};
use std::{marker::PhantomData, ops::Index};

pub struct View<'a, Tn: Tensor> {
    _marker: PhantomData<Tn>,
    storage: &'a [Tn::T],
    idx_translate: Box<dyn Fn(&Tn::Idx) -> usize>,
}

impl<'a, Tn: Tensor> View<'a, Tn> {
    pub(crate) fn new(storage: &'a [Tn::T], idx_translate: Box<dyn Fn(&Tn::Idx) -> usize>) -> Self {
        Self {
            _marker: PhantomData,
            storage,
            idx_translate,
        }
    }
}

impl<'a, Tn: Tensor> View<'a, Tn>
where
    Tn: ShapedTensor,
    Tn::Idx: for<'b> From<&'b [usize]>,
{
    pub fn reduce_dim<const DIM: usize>(
        self,
        f: impl Fn(Tn::T, Tn::T) -> Tn::T + 'static,
    ) -> GenericTensor<Tn::T, { Tn::R }, { reduced_shape(Tn::R, Tn::S, DIM) }> {
        GenericTensor::from_fn(|idx| {
            let mut src_idx = *idx;
            debug_assert!(src_idx[DIM] == 0);

            let mut res = self[&src_idx[..].into()];
            for i in 1..Tn::S[DIM] {
                src_idx[DIM] = i;
                res = f(res, self[&src_idx[..].into()]);
            }
            res
        })
    }
}

impl<'a, 'b, Tn: Tensor> Index<&'b Tn::Idx> for View<'a, Tn> {
    type Output = Tn::T;

    fn index(&self, idx: &'b Tn::Idx) -> &Self::Output {
        let t = self.idx_translate.as_ref();
        let i = t(idx);
        self.storage.index(i)
    }
}
