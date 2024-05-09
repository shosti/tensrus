use crate::{
    generic_tensor::GenericTensor,
    numeric::Numeric,
    scalar::scalar_shape,
    tensor::{stride, IndexError, Shape},
};

#[derive(Debug, PartialEq, Eq)]
pub struct Slice<'a, T: Numeric, const R: usize, const S: Shape> {
    storage: &'a Vec<T>,
    offset: usize,
}

impl<'a, T: Numeric, const R: usize, const S: Shape> Slice<'a, T, R, S> {
    pub fn new<const D: usize, const R2: usize, const S2: Shape>(
        storage: &'a Vec<T>,
        idx: [usize; D],
    ) -> Result<Self, IndexError> {
        for dim in 0..D {
            if idx[dim] >= S2[dim] {
                return Err(IndexError {});
            }
        }

        let mut offset = 0;
        let str = stride::<R2, S2>();
        for i in 0..D {
            offset += idx[i] * str[i];
        }

        Ok(Self { storage, offset })
    }
}

impl<'a, T: Numeric> Slice<'a, T, 0, { scalar_shape() }> {
    pub fn val(&self) -> T {
        self.storage[self.offset]
    }
}

impl<'a, T: Numeric, const R: usize, const S: Shape> Into<GenericTensor<T, R, S>>
    for Slice<'a, T, R, S>
{
    fn into(self) -> GenericTensor<T, R, S> {
        self.storage[self.offset..].iter().map(|x| *x).collect()
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::tensor::Tensor;

//     #[test]
//     fn test_slice_and_back() {
//         let t: GenericTensor<f64, { Shape::Rank1([5]) }> = GenericTensor::from([1, 2, 3, 4, 5]);
//         let s = t.slice([]);
//         let t2: GenericTensor<f64, { Shape::Rank1([5]) }> = s.into();
//         assert_eq!(t, t2);
//     }

//     #[test]
//     fn test_slice_to_val() {
//         let t: GenericTensor<f64, { Shape::Rank1([5]) }> = GenericTensor::from([0, 1, 2, 3, 4]);
//         for i in 0..5 {
//             let s = t.slice([i]);
//             assert_eq!(s.val(), i as f64);
//         }
//     }

//     #[test]
//     fn test_slice_rank2() {
//         #[rustfmt::skip]
//         let t: GenericTensor<f64, { Shape::Rank2([3, 2]) }> = GenericTensor::from([
//             1, 2,
//             3, 4,
//             5, 6,
//         ]);

//         assert_eq!(t.slice([2, 1]).val(), 6.0);
//         assert_eq!(t.slice([0, 0]).val(), 1.0);
//         assert_eq!(t.slice([1, 0]).val(), 3.0);

//         let subtensor: GenericTensor<f64, { Shape::Rank1([2]) }> = t.slice([1]).into();
//         let want: GenericTensor<f64, { Shape::Rank1([2]) }> = GenericTensor::from([3, 4]);

//         assert_eq!(subtensor, want);
//     }

//     #[test]
//     fn test_try_slice() {
//         #[rustfmt::skip]
//         let t: GenericTensor<f64, { Shape::Rank2([3, 2]) }> = GenericTensor::from([
//             1, 2,
//             3, 4,
//             5, 6,
//         ]);

//         assert_eq!(t.try_slice([2, 1]).unwrap().val(), 6.0);
//         assert_eq!(t.try_slice([3, 0]), Err(IndexError {}));
//         assert_eq!(t.try_slice([0, 2]), Err(IndexError {}));
//     }
// }
