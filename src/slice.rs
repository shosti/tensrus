use crate::{
    generic_tensor::GenericTensor,
    numeric::Numeric,
    shape::Shape,
    type_assert::{Assert, IsTrue},
};

pub struct Slice<'a, T: Numeric, const S: Shape> {
    storage: &'a Vec<T>,
    offset: usize,
}

impl<'a, T: Numeric, const S: Shape> Slice<'a, T, S> {
    pub fn new<const D: usize>(
        storage: &'a Vec<T>,
        idx: [usize; D],
    ) -> Self {
        let mut offset = 0;
        for i in 0..D {
            offset += idx[i] * S.stride()[i];
        }
        Self { storage, offset }
    }
}

impl<'a, T: Numeric> Slice<'a, T, { Shape::Rank0([]) }> {
    fn val(&self) -> T {
        self.storage[self.offset]
    }
}

impl<'a, T: Numeric, const S: Shape> Into<GenericTensor<T, S>> for Slice<'a, T, S> {
    fn into(self) -> GenericTensor<T, S> {
        self.storage[self.offset..].iter().map(|x| *x).collect()
    }
}
