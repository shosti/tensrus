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
        let mut offset = 0;
        for i in 0..(S.downrank(D).rank()) {
            offset += idx[i] * S.stride()[i];
        }
        Self { storage, offset }
    }
}
