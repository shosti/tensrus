use crate::tensor::{IndexError, Storage, Tensor};
use num::Num;
use std::ops::MulAssign;

#[derive(Debug)]
pub struct Vector<T: Num + Copy, const N: usize> {
    vals: Storage<T, N>,
}

impl<T: Num + Copy, const N: usize> From<[T; N]> for Vector<T, N> {
    fn from(vals: [T; N]) -> Self {
        Vector {
            vals: Storage::from(vals),
        }
    }
}

impl<T: Num + Copy, const N: usize> Tensor<T, 1> for Vector<T, N> {
    fn shape(&self) -> [usize; 1] {
        [N]
    }

    fn get(&self, idx: [usize; 1]) -> Result<T, IndexError> {
        let [i] = idx;
        if i >= N {
            return Err(IndexError {});
        }

        Ok(self.vals.get(idx[0]))
    }

    fn set(&mut self, idx: [usize; 1], val: T) -> Result<(), IndexError> {
        let [i] = idx;
        if i >= N {
            return Err(IndexError {});
        }
        self.vals.set(idx[0], val);

        Ok(())
    }
}

impl<T: Num + Copy, const N: usize> MulAssign<T> for Vector<T, N> {
    fn mul_assign(&mut self, other: T) {
        self.vals.elem_mul(other);
    }
}
