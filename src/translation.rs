use crate::{
    generic_tensor::{GenericTensor, IntoGeneric},
    iterator::Iter,
    shape::{subtensor_shape, Broadcastable, Reducible, Shape, Shaped, Transposable},
    storage::{Layout, Storage, TensorLayout},
    tensor::{Indexable, Tensor, TensorIndex},
    view::View,
};
use std::ops::Index;

pub struct Translation<'a, Tn: Tensor> {
    storage: &'a Storage<Tn::T>,
    idx_translate: Box<dyn Fn(Tn::Idx) -> usize + 'a>,
}

impl<'a, Tn: Tensor> Translation<'a, Tn> {
    pub fn new<Src>(
        storage: &'a Storage<Tn::T>,
        f: impl Fn(Tn::Idx) -> Src::Idx + 'a,
    ) -> Translation<'a, Tn>
    where
        Src: Tensor<T = Tn::T>,
    {
        let idx_translate = Box::new(move |idx: Tn::Idx| {
            let src_idx = f(idx);

            View::<Src>::calc_storage_idx(&src_idx, storage.layout)
        });

        Translation {
            storage,
            idx_translate,
        }
    }

    pub fn translate<Dest>(self, f: impl Fn(Dest::Idx) -> Tn::Idx + 'a) -> Translation<'a, Dest>
    where
        Dest: Tensor<T = Tn::T>,
    {
        let idx_translate = Box::new(move |dest_idx: Dest::Idx| {
            let src_idx = f(dest_idx);

            (self.idx_translate)(src_idx)
        });

        Translation {
            storage: self.storage,
            idx_translate,
        }
    }

    pub fn iter(&self) -> Iter<Self> {
        Iter::new(self)
    }

    pub fn to_owned(&self) -> Tn
    where
        Tn::T: Clone,
    {
        Tn::from_fn(|idx| self[idx].clone())
    }

    pub fn into_generic<const R: usize, const S: Shape>(
        self,
    ) -> Translation<'a, GenericTensor<Tn::T, R, S>>
    where
        Tn: IntoGeneric<Tn::T, R, S>,
    {
        let idx_translate = Box::new(move |idx: [usize; R]| {
            let src_idx = Tn::Idx::from_slice(idx.as_ref());
            (self.idx_translate)(src_idx)
        });

        Translation {
            storage: self.storage,
            idx_translate,
        }
    }

    fn storage_idx(&self, idx: &Tn::Idx) -> usize {
        (self.idx_translate)(*idx)
    }

    pub fn reduce_dim<const DIM: usize>(
        self,
        f: impl Fn(Tn::T, Tn::T) -> Tn::T + 'static,
    ) -> Tn::Reduced
    where
        Tn: Reducible<DIM>,
        Tn::T: Copy,
    {
        Tn::Reduced::from_fn(|idx| {
            let mut src_idx = *idx;
            debug_assert!(src_idx[DIM] == 0);

            let mut res = self[idx];
            for i in 1..Tn::shape()[DIM] {
                src_idx[DIM] = i;
                res = f(res, self[&src_idx]);
            }
            res
        })
    }

    pub fn broadcast<Dest>(self) -> Translation<'a, Dest>
    where
        Tn: Broadcastable<Dest>,
        Dest: Tensor<T = Tn::T>,
    {
        let layout = self.storage.layout;
        let idx_translate = Box::new(move |bcast_idx: Dest::Idx| {
            let src_idx = Tn::unbroadcasted_idx(&bcast_idx);

            crate::storage::storage_idx(src_idx.as_ref(), Tn::rank(), Tn::shape(), layout).unwrap()
        });
        Translation {
            storage: self.storage,
            idx_translate,
        }
    }
}

impl<'a, T, const R: usize, const S: Shape> Translation<'a, GenericTensor<T, R, S>> {
    pub fn from_generic<Tn>(self) -> Translation<'a, Tn>
    where
        Tn: Tensor<T = T> + IntoGeneric<T, R, S>,
    {
        let idx_translate = Box::new(move |idx: Tn::Idx| {
            let src_idx = <[usize; R]>::from_slice(idx.as_ref());
            (self.idx_translate)(src_idx)
        });
        Translation {
            storage: self.storage,
            idx_translate,
        }
    }

    pub fn subtensor<'b: 'a, const N: usize>(
        self,
        idx: &'b [usize; N],
    ) -> Translation<'a, GenericTensor<T, { R - N }, { subtensor_shape(N, R, S) }>> {
        let src_translate = self.idx_translate;
        let idx_translate = Box::new(move |dest_idx: [usize; R - N]| {
            let mut src_idx = [0; R];
            src_idx[0..N].copy_from_slice(&idx[..]);
            src_idx[N..R].copy_from_slice(&dest_idx);

            src_translate(src_idx)
        });

        Translation {
            storage: self.storage,
            idx_translate,
        }
    }
}

impl<'a, Tn> Shaped for Translation<'a, Tn>
where
    Tn: Tensor,
{
    fn shape() -> Shape {
        Tn::shape()
    }

    fn rank() -> usize {
        Tn::rank()
    }
}

impl<'a, Tn, Dest> Transposable<Translation<'a, Dest>> for Translation<'a, Tn>
where
    Tn: Tensor + Transposable<Dest>,
    Dest: Tensor<T = Tn::T>,
{
    fn transpose(self) -> Translation<'a, Dest> {
        self.translate(move |idx: Dest::Idx| {
            let src_idx = Tn::Idx::from_slice(idx.as_ref());
            src_idx.transpose()
        })
    }
}

impl<'a, 'b, Tn> Index<&'b Tn::Idx> for Translation<'a, Tn>
where
    Tn: Tensor,
{
    type Output = Tn::T;

    fn index(&self, idx: &'b Tn::Idx) -> &Self::Output {
        self.storage.data.index(self.storage_idx(idx))
    }
}

impl<'a, Tn, Rhs> PartialEq<Rhs> for Translation<'a, Tn>
where
    Tn: Tensor,
    Rhs: Indexable<T = Tn::T, Idx = Tn::Idx>,
    Tn::T: PartialEq,
{
    fn eq(&self, other: &Rhs) -> bool {
        self.iter().all(|(idx, val)| val == other.index(&idx))
    }
}

impl<'a, Tn: Tensor> Eq for Translation<'a, Tn> where Tn::T: Eq {}

impl<'a, Tn: Tensor> std::fmt::Debug for Translation<'a, Tn>
where
    Tn::T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        f.debug_struct("Translation ")
            .field("storage", self.storage)
            .finish()
    }
}

impl<'a, Tn> TensorLayout for Translation<'a, Tn>
where
    Tn: Tensor,
{
    fn layout(&self) -> Layout {
        self.storage.layout
    }
}

impl<'a, Tn> Indexable for Translation<'a, Tn>
where
    Tn: Tensor,
{
    type Idx = Tn::Idx;
    type T = Tn::T;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        matrix::Matrix,
        scalar::Scalar,
        shape::{self, Transposable},
        vector::Vector,
    };

    #[test]
    fn test_reduce_dim() {
        let t = GenericTensor::<_, 2, { shape::rank2([2, 3]) }>::from([1, 2, 3, 4, 5, 6]);
        let v = t.view();
        let t2 = v.reduce_dim::<0>(|x, y| x + y);
        assert_eq!(
            t2,
            GenericTensor::<_, 2, { shape::rank2([1, 3]) }>::from([5, 7, 9])
        );
        let t3: GenericTensor<_, 2, { shape::rank2([2, 1]) }> =
            t.view().reduce_dim::<1>(|x, y| x + y);
        assert_eq!(
            t3,
            GenericTensor::<_, 2, { shape::rank2([2, 1]) }>::from([6, 15])
        );
    }

    #[test]
    fn test_broadcast() {
        let x = Vector::from([1, 2, 3]);
        let v = x.as_generic();
        let t: Translation<Matrix<_, 3, 3>> = v.broadcast().from_generic();
        let m = t.to_owned();

        assert_eq!(m, Matrix::from([[1, 2, 3], [1, 2, 3], [1, 2, 3]]));

        let t: GenericTensor<_, 3, { shape::rank3([3, 3, 3]) }> =
            x.as_generic().broadcast().to_owned();
        assert_eq!(
            t,
            [1, 2, 3]
                .into_iter()
                .cycle()
                .collect::<GenericTensor<_, 3, { shape::rank3([3, 3, 3]) }>>()
        );

        let x = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let one = Scalar::from(1);
        let one_gen = one.as_generic();
        let ones: Translation<Matrix<_, 2, 3>> = one_gen.broadcast().from_generic();

        assert_eq!(x + ones, Matrix::from([[2, 3, 4], [5, 6, 7]]));
    }

    #[test]
    fn test_subtensor() {
        let t: GenericTensor<_, 3, { shape::rank3([2, 3, 4]) }> = (0..24).collect();
        assert_eq!(t.view().subtensor(&[]), t.view());

        let sub_1 = t.view().subtensor(&[1]);
        let want_1: GenericTensor<_, 2, { shape::rank2([3, 4]) }> = (12..24).collect();
        assert_eq!(sub_1, want_1.view());

        let sub_2 = t.view().subtensor(&[1, 2]);
        let want_2: GenericTensor<_, 1, { shape::rank1([4]) }> = (20..24).collect();
        assert_eq!(sub_2, want_2.view());

        let sub_3 = t.view().subtensor(&[1, 2, 3]);
        let want_3: GenericTensor<_, 0, { shape::rank0() }> = GenericTensor::from([23]);
        assert_eq!(sub_3, want_3.view());

        // Test transposed version
        let t_t: GenericTensor<_, 3, { shape::rank3([2, 3, 4]) }> = [
            // Ends up as 0..24 once transposed
            0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23,
        ]
        .into_iter()
        .collect::<GenericTensor<_, 3, { shape::rank3([4, 3, 2]) }>>()
        .transpose();
        assert_eq!(t_t.view().subtensor(&[]), t.view());

        let sub_1_t = t_t.view().subtensor(&[1]);
        assert_eq!(sub_1_t, want_1.view());

        let sub_2_t = t_t.view().subtensor(&[1, 2]);
        assert_eq!(sub_2_t, want_2.view());

        let sub_3_t = t_t.view().subtensor(&[1, 2, 3]);
        assert_eq!(sub_3_t, want_3.view());
    }
}
