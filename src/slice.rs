use crate::{
    numeric::Numeric,
    shape::Shape,
    type_assert::{Assert, IsTrue},
};

struct Slice<'a, T: Numeric, const S: Shape> {
    storage: &'a Vec<T>,
    offset: usize,
}

impl<'a, T: Numeric, const S: Shape> Slice<'a, T, S> {
    pub fn from_idx<const D: usize>(
        storage: &'a Vec<T>,
        idx: [usize; S.downrank(D).rank()],
    ) -> Self {
        let res = Self::from_idx_with_offset(storage, idx, 0);
        Self {
            storage: res.storage,
            offset: res.offset,
        }
    }

    fn from_idx_with_offset<const D: usize>(
        storage: &'a Vec<T>,
        idx: [usize; S.downrank(D).rank()],
        offset: usize,
    ) -> Slice<'a, T, { S.downrank(D) }> {
        todo!()
    }
}
