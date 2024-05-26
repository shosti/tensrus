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

#[cfg(test)]
mod tests {
    use crate::{
        generic_tensor::GenericTensor,
        matrix::Matrix,
        tensor::{IndexError, SlicedTensor},
        vector::Vector,
    };

    #[test]
    fn test_slice_and_back() {
        let v = Vector::<f64, _>::from([1, 2, 3, 4, 5]);
        let s = v.slice([]);
        let v2: Vector<f64, _> = s.into();
        assert_eq!(v, v2);
    }

    #[test]
    fn test_slice_to_val() {
        let t: Vector<f64, _> = Vector::from([0, 1, 2, 3, 4]);
        for i in 0..5 {
            let s = t.slice([i]);
            assert_eq!(s.val(), i as f64);
        }
    }

    #[test]
    fn test_slice_matrix() {
        #[rustfmt::skip]
        let t: Matrix<f64, _, _> = Matrix::from([
            [1, 2],
            [3, 4],
            [5, 6],
        ]);

        assert_eq!(t.slice([2, 1]).val(), 6.0);
        assert_eq!(t.slice([0, 0]).val(), 1.0);
        assert_eq!(t.slice([1, 0]).val(), 3.0);

        let vector: Vector<f64, _> = t.slice([1]).into();
        let want: Vector<f64, _> = Vector::from([3, 4]);

        assert_eq!(vector, want);
    }

    // TODO: Think about how this should be implemented
    #[test]
    #[should_panic]
    fn test_matrix_transpose() {
        #[rustfmt::skip]
        let t: Matrix<f64, _, _> = Matrix::from([
            [1, 2],
            [3, 4],
            [5, 6],
        ]).transpose();
        assert_eq!(t.slice([0, 2]).val(), 5.0);

        let t2: Matrix<f64, 2, 3> = t.slice([]).into();
        #[rustfmt::skip]
        let want: Matrix<f64, _, _> = Matrix::from([
            [1, 3, 5],
            [2, 4, 6],
        ]);
        assert_eq!(t2, want);
    }

    #[test]
    fn test_try_slice() {
        #[rustfmt::skip]
        let t: GenericTensor<f64, 2, { [3, 2, 0, 0, 0] }> = GenericTensor::from([
            1, 2,
            3, 4,
            5, 6,
        ]);

        assert_eq!(t.try_slice([2, 1]).unwrap().val(), 6.0);
        assert_eq!(t.try_slice([3, 0]), Err(IndexError {}));
        assert_eq!(t.try_slice([0, 2]), Err(IndexError {}));
    }
}
