use crate::{
    broadcast::{broadcast_compat, broadcast_normalize},
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
    idx_translate: Option<Box<dyn Fn(Tn::Idx) -> usize>>,
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

pub trait Broadcastable: Tensor + ShapedTensor + TensorStorage<Self::T> {
    fn broadcast<Dest>(&self) -> View<Dest>
    where
        Dest: Tensor<T = Self::T> + ShapedTensor,
        Assert<{ broadcast_compat(Self::R, Self::S, Dest::R, Dest::S) }>: IsTrue,
    {
        let layout = self.layout();
        let t = Box::new(move |dest_idx: Dest::Idx| {
            let idx: &[usize] = dest_idx.as_ref();
            let s_normalized = broadcast_normalize(Self::S, Self::R, Dest::R);

            let mut src_idx = [0; Self::R];
            let mut dim = 0;
            for i in 0..Dest::R {
                if s_normalized[i] == 1 && Dest::S[i] != 1 {
                    continue;
                }
                src_idx[dim] = idx[i];
                dim += 1;
            }

            crate::storage::storage_idx_gen(Self::R, &src_idx, Self::S, layout).unwrap()
        });
        View {
            storage: self.storage(),
            layout,
            idx_translate: Some(t),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{matrix::Matrix, shape::MAX_DIMS, vector::Vector};

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
        let t3 = t.view().reduce_dim::<1>(|x, y| x + y);
        assert_eq!(
            t3,
            GenericTensor::<f64, 2, { [2, 1, 0, 0, 0, 0] }>::from([6, 15])
        );
    }

    #[test]
    fn test_broadcast() {
        let v = Vector::<f64, _>::from([1, 2, 3]);
        let m: Matrix<_, 3, 3> = v.broadcast().into();
        assert_eq!(
            m,
            Matrix::<f64, _, _>::from([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        );

        let t: GenericTensor<_, 3, { [3; MAX_DIMS] }> = v.broadcast().into();
        assert_eq!(
            t,
            [1, 2, 3]
                .into_iter()
                .cycle()
                .collect::<GenericTensor<f64, 3, { [3; MAX_DIMS] }>>()
        );
    }
}
