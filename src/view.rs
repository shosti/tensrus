use crate::{
    generic_tensor::GenericTensor,
    shape::reduced_shape,
    storage::{Layout, TensorStorage},
    tensor::{ShapedTensor, Tensor},
};
use std::ops::Index;

pub struct View<'a, Tn: Tensor> {
    storage: &'a [Tn::T],
    layout: Layout,
    idx_translate: Option<Box<dyn Fn(Tn::Idx) -> usize>>,
}

impl<'a, Tn: Tensor> View<'a, Tn> {
    pub fn with_translation(
        storage: &'a [Tn::T],
        layout: Layout,
        t: Box<dyn Fn(Tn::Idx) -> usize>,
    ) -> Self {
        Self {
            storage,
            layout,
            idx_translate: Some(t),
        }
    }
}

impl<'a, Tn: Tensor> View<'a, Tn>
where
    Tn: ShapedTensor,
    Tn::Idx: From<[usize; Tn::R]>,
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

impl<'a, Tn> From<View<'a, GenericTensor<Tn::T, { Tn::R }, { Tn::S }>>> for View<'a, Tn>
where
    Tn: ShapedTensor + Tensor + From<GenericTensor<Tn::T, { Tn::R }, { Tn::S }>>,
    [usize; Tn::R]: From<Tn::Idx>,
{
    fn from(v: View<'a, GenericTensor<Tn::T, { Tn::R }, { Tn::S }>>) -> Self {
        let storage = v.storage;
        let layout = v.layout;
        match v.idx_translate {
            None => Self {
                storage,
                layout,
                idx_translate: None,
            },
            Some(t) => {
                let idx_translate = Box::new(move |idx_orig: Tn::Idx| {
                    let idx: [usize; Tn::R] = idx_orig.into();
                    t(idx)
                });
                Self {
                    storage,
                    layout,
                    idx_translate: Some(idx_translate),
                }
            }
        }
    }
}

impl<'a, 'b, Tn> Index<&'b Tn::Idx> for View<'a, Tn>
where
    Tn: Tensor + ShapedTensor,
{
    type Output = Tn::T;

    fn index(&self, idx: &'b Tn::Idx) -> &Self::Output {
        let i = match &self.idx_translate {
            Some(t) => t(*idx),
            None => crate::storage::storage_idx_gen(Tn::R, idx.as_ref(), Tn::S, self.layout)
                .expect("out of bounds"),
        };
        self.storage.index(i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_dim() {
        let t = GenericTensor::<f64, 2, { [2, 3, 0, 0, 0, 0] }>::from([1, 2, 3, 4, 5, 6]);
        let v: View<_> = (&t).into();
        let t2 = v.reduce_dim::<0>(|x, y| x + y);
        assert_eq!(
            t2,
            GenericTensor::<f64, 2, { [1, 3, 0, 0, 0, 0] }>::from([5, 7, 9])
        );
        let t3: GenericTensor<f64, 2, { [2, 1, 0, 0, 0, 0] }> =
            t.view().reduce_dim::<1>(|x, y| x + y);
        assert_eq!(
            t3,
            GenericTensor::<f64, 2, { [2, 1, 0, 0, 0, 0] }>::from([6, 15])
        );
    }
}
