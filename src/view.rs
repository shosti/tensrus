use crate::{
    broadcast::broadcast_normalize,
    generic_tensor::GenericTensor,
    numeric::Numeric,
    shape::{reduced_shape, shapes_equal, Shape, Shaped, MAX_DIMS},
    storage::Layout,
    tensor::{Indexable, Tensor, TensorIndex, TensorIterator},
};
use std::ops::Index;

pub struct View<'a, Tn: Tensor> {
    storage: &'a [Tn::T],
    layout: Layout,
    idx_translate: Option<Box<dyn Fn(Tn::Idx) -> usize>>,
}

impl<'a, Tn: Tensor> View<'a, Tn> {
    pub fn new(storage: &'a [Tn::T], layout: Layout) -> Self {
        Self {
            storage,
            layout,
            idx_translate: None,
        }
    }

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

    pub fn iter(&self) -> TensorIterator<Self> {
        TensorIterator::new(self)
    }

    pub fn to_generic(self) -> View<'a, GenericTensor<Tn::T, { Tn::R }, { Tn::S }>> {
        match self.idx_translate {
            None => View {
                storage: self.storage,
                layout: self.layout,
                idx_translate: None,
            },
            Some(t) => {
                let tr = Box::new(move |idx: [usize; Tn::R]| {
                    let idx = Tn::Idx::from_slice(&idx);
                    t(idx)
                });

                View {
                    storage: self.storage,
                    layout: self.layout,
                    idx_translate: Some(tr),
                }
            }
        }
    }
}

impl<'a, T: Numeric, const R: usize, const S: Shape> View<'a, GenericTensor<T, R, S>> {
    pub fn from_generic<Tn>(self) -> View<'a, Tn>
    where
        Tn: Tensor<T = T> + From<GenericTensor<T, R, S>>,
    {
        match self.idx_translate {
            None => View {
                storage: self.storage,
                layout: self.layout,
                idx_translate: None,
            },
            Some(t) => {
                let tr = Box::new(move |tn_idx: Tn::Idx| {
                    let mut idx = [0; R];
                    idx.copy_from_slice(tn_idx.as_ref());
                    t(idx)
                });

                View {
                    storage: self.storage,
                    layout: self.layout,
                    idx_translate: Some(tr),
                }
            }
        }
    }

    pub fn reduce_dim<const DIM: usize>(
        self,
        f: impl Fn(T, T) -> T + 'static,
    ) -> GenericTensor<T, R, { reduced_shape(R, S, DIM) }> {
        GenericTensor::from_fn(|idx| {
            let mut src_idx: [usize; R] = *idx;
            debug_assert!(src_idx[DIM] == 0);

            let mut res = self[idx];
            for i in 1..S[DIM] {
                src_idx[DIM] = i;
                res = f(res, self[&src_idx]);
            }
            res
        })
    }

    pub fn broadcast<const R_DEST: usize, const S_DEST: Shape>(
        &self,
    ) -> View<GenericTensor<T, R_DEST, S_DEST>> {
        if shapes_equal(R, S, R_DEST, S_DEST) {
            return View::new(self.storage, self.layout);
        }
        let layout = self.layout;
        let t = Box::new(move |bcast_idx: [usize; R_DEST]| {
            let src_idx = Self::unbroadcasted_idx::<R_DEST, S_DEST>(&bcast_idx);

            crate::storage::storage_idx(R, &src_idx, S, layout).unwrap()
        });
        View::with_translation(self.storage, layout, t)
    }

    pub fn unbroadcasted_idx<const R_DEST: usize, const S_DEST: Shape>(
        bcast_idx: &[usize; R_DEST],
    ) -> [usize; R] {
        let s_normalized = broadcast_normalize(S, R, R_DEST);

        let mut src_idx = [0; R];
        let mut dim = 0;
        for i in 0..R_DEST {
            if s_normalized[i] == 1 && S_DEST[i] != 1 {
                continue;
            }
            src_idx[dim] = bcast_idx[i];
            dim += 1;
        }

        src_idx
    }
}

impl<'a, Tn: Tensor> Indexable for View<'a, Tn> {
    type Idx = Tn::Idx;
    type T = Tn::T;

    fn num_elems() -> usize {
        Tn::num_elems()
    }
    fn default_idx() -> Self::Idx {
        Tn::default_idx()
    }
    fn next_idx(&self, idx: &Self::Idx) -> Option<Self::Idx> {
        let shape = Tn::shape();
        let i = crate::storage::storage_idx(Self::R, idx.as_ref(), shape, self.layout).ok()?;
        if i >= Self::num_elems() - 1 {
            return None;
        }

        let mut idx = [0; MAX_DIMS];
        crate::storage::get_nth_idx(i + 1, &mut idx, Self::R, shape, self.layout).ok()?;
        Some(Self::Idx::from_slice(&idx))
    }
}

impl<'a, Tn> From<&'a Tn> for View<'a, Tn>
where
    Tn: Tensor,
{
    fn from(t: &'a Tn) -> Self {
        Self {
            storage: t.storage(),
            layout: t.layout(),
            idx_translate: None,
        }
    }
}

impl<'a, Tn: Tensor + Shaped> Shaped for View<'a, Tn> {
    const R: usize = Tn::R;
    const S: Shape = Tn::S;
}

impl<'a, Tn> From<View<'a, GenericTensor<Tn::T, { Tn::R }, { Tn::S }>>> for View<'a, Tn>
where
    Tn: Tensor + From<GenericTensor<Tn::T, { Tn::R }, { Tn::S }>>,
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
    Tn: Tensor,
{
    type Output = Tn::T;

    fn index(&self, idx: &'b Tn::Idx) -> &Self::Output {
        let i = match &self.idx_translate {
            Some(t) => t(*idx),
            None => crate::storage::storage_idx(Tn::R, idx.as_ref(), Tn::S, self.layout)
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
