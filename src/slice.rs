use crate::{
    generic_tensor::GenericTensor,
    numeric::Numeric,
    scalar::scalar_shape,
    tensor::{stride, IndexError, Shape, Transpose},
};

#[derive(Debug, PartialEq, Eq)]
pub struct Slice<'a, T: Numeric, const R: usize, const S: Shape> {
    storage: &'a Vec<T>,
    offset: usize,
    transpose: Transpose,
}

impl<'a, T: Numeric, const R: usize, const S: Shape> Slice<'a, T, R, S> {
    pub fn new<const D: usize, const R2: usize, const S2: Shape>(
        storage: &'a Vec<T>,
        transpose: Transpose,
        idx: [usize; D],
    ) -> Result<Self, IndexError> {
        for (i, &dim) in idx.iter().enumerate().take(D) {
            if dim >= S2[i] {
                return Err(IndexError {});
            }
        }

        if transpose == Transpose::Transposed {
            panic!("slice of transposed tensor isn't implemented yet");
        }

        let mut offset = 0;
        let str = stride(R2, S2);
        for i in 0..D {
            offset += idx[i] * str[i];
        }

        Ok(Self {
            storage,
            offset,
            transpose,
        })
    }
}

impl<'a, T: Numeric> Slice<'a, T, 0, { scalar_shape() }> {
    pub fn val(&self) -> T {
        self.storage[self.offset]
    }
}

impl<'a, T: Numeric, const R: usize, const S: Shape> From<Slice<'a, T, R, S>>
    for GenericTensor<T, R, S>
{
    fn from(s: Slice<'a, T, R, S>) -> Self {
        let storage = s.storage[s.offset..].to_vec();
        Self {
            storage,
            transpose_state: s.transpose,
        }
    }
}
