use crate::numeric::Numeric;

#[derive(Debug, PartialEq)]
pub struct IndexError {}

pub struct Tensor<T: Numeric, const R: usize, const S: [usize; 5]> {
    _vals: Vec<T>,
}

impl<T: Numeric, const R: usize, const S: [usize; 5]> Tensor<T, R, S> {
    pub fn zeros() -> Self {
        let vals = vec![T::zero(); Self::storage_size()];
        Self { _vals: vals }
    }

    pub fn rank(&self) -> usize {
        R
    }
    pub fn shape(&self) -> [usize; R] {
        let mut s = [0; R];
        for i in 0..R {
            s[i] = S[i];
        }

        s
    }

    fn storage_size() -> usize {
        let mut size = 1;
        for i in 0..R {
            size *= S[i];
        }
        size
    }
}

// pub trait Tensor<T: Numeric, const R: usize>: MulAssign<T> + Eq {
//     type Transpose;

//     fn from_fn<F>(cb: F) -> Self
//     where
//         F: FnMut([usize; R]) -> T;
//     fn zeros() -> Self
//     where
//         Self: Sized,
//     {
//         Self::from_fn(|_| T::zero())
//     }

//     fn rank(&self) -> usize {
//         R
//     }
//     fn shape(&self) -> [usize; R];
//     fn get(&self, idx: [usize; R]) -> Result<T, IndexError>;
//     fn set(&mut self, idx: [usize; R], val: T) -> Result<(), IndexError>;
//     fn transpose(&self) -> Self::Transpose;
//     fn next_idx(&self, idx: [usize; R]) -> Option<[usize; R]>;

//     fn update<F>(&mut self, mut cb: F)
//     where
//         F: FnMut(T) -> T,
//     {
//         let mut iter = Some([0; R]);
//         while let Some(idx) = iter {
//             let cur = self.get(idx).unwrap();
//             self.set(idx, cb(cur)).unwrap();
//             iter = self.next_idx(idx);
//         }
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_and_shape() {
        let scalar: Tensor<f64, 0, { [0; 5] }> = Tensor::zeros();

        assert_eq!(scalar.rank(), 0);
        assert_eq!(scalar.shape(), []);

        let vector: Tensor<f64, 1, { [7; 5] }> = Tensor::zeros();

        assert_eq!(vector.rank(), 1);
        assert_eq!(vector.shape(), [7]);

        let matrix: Tensor<f64, 2, { [3, 2, 0, 0, 0] }> = Tensor::zeros();

        assert_eq!(matrix.rank(), 2);
        assert_eq!(matrix.shape(), [3, 2]);

        let tensor3: Tensor<f64, 3, { [7, 8, 2, 0, 0] }> = Tensor::zeros();
        assert_eq!(tensor3.rank(), 3);
        assert_eq!(tensor3.shape(), [7, 8, 2]);
    }
}
