use crate::{
    broadcast::broadcast_compat,
    generic_tensor::GenericTensor,
    shape::reduced_shape,
    storage::{Layout, TensorStorage},
    tensor::{ShapedTensor, Tensor},
    type_assert::{Assert, IsTrue},
};
use std::ops::Index;

pub struct View<'a, Tn: Tensor> {
    storage: &'a [Tn::T],
    layout: Layout,
    idx_translate: Option<Box<dyn Fn(&Tn::Idx, Layout) -> usize>>,
}

impl<'a, Tn: Tensor> View<'a, Tn>
where
    Tn: ShapedTensor,
    Tn::Idx: From<[usize; Tn::R]>,
    Tn::Idx: AsRef<[usize; Tn::R]>,
{
    pub fn reduce_dim<const DIM: usize>(
        self,
        f: impl Fn(Tn::T, Tn::T) -> Tn::T + 'static,
    ) -> GenericTensor<Tn::T, { Tn::R }, { reduced_shape(Tn::R, Tn::S, DIM) }> {
        GenericTensor::from_fn(|idx| {
            let mut src_idx = *idx;
            debug_assert!(src_idx[DIM] == 0);

            let mut res = self[&src_idx.into()];
            for i in 1..Tn::S[DIM] {
                src_idx[DIM] = i;
                res = f(res, self[&src_idx.into()]);
            }
            res
        })
    }
}

impl<'a, Tn> From<&'a Tn> for View<'a, Tn>
where
    Tn: Tensor + TensorStorage<Tn::T>,
{
    fn from(t: &'a Tn) -> Self {
        Self {
            storage: t.storage(),
            layout: t.layout(),
            idx_translate: None,
        }
    }
}

impl<'a, 'b, Tn> Index<&'b Tn::Idx> for View<'a, Tn>
where
    Tn: Tensor + ShapedTensor,
    Tn::Idx: AsRef<[usize; Tn::R]>,
{
    type Output = Tn::T;

    fn index(&self, idx: &'b Tn::Idx) -> &Self::Output {
        let i = match &self.idx_translate {
            Some(t) => t.as_ref()(idx, self.layout),
            None => crate::storage::storage_idx::<{ Tn::R }>(idx.as_ref(), Tn::S, self.layout)
                .expect("out of bounds"),
        };
        self.storage.index(i)
    }
}

pub trait Broadcastable: Tensor + ShapedTensor {
    fn broadcast<Dest>(&self) -> View<Dest>
    where
        Dest: Tensor<T = Self::T> + ShapedTensor,
        Assert<{ broadcast_compat(Self::R, Self::S, Dest::R, Dest::S) }>: IsTrue,
    {
        todo!()
    }
}
