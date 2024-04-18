use crate::tensor::{IndexError, Storage, Tensor};
use num::Num;
use std::ops::MulAssign;

#[derive(Debug)]
pub struct Vector<T: Num + Copy, const N: usize> {
    vals: Storage<T, N>,
}

impl<T: Num + Copy, const N: usize> Vector<T, N> {
    pub fn dot(&self, other: &Self) -> T {
        let mut res = T::zero();
        for i in 0..N {
            res = res + (self.get([i]).unwrap() * other.get([i]).unwrap());
        }

        res
    }
}

impl<T: Num + Copy, const N: usize> From<[T; N]> for Vector<T, N> {
    fn from(vals: [T; N]) -> Self {
        Vector {
            vals: Storage::from(vals),
        }
    }
}

impl<T: Num + Copy, const N: usize> Tensor<T, 1> for Vector<T, N> {
    fn from_fn<F>(mut cb: F) -> Self
    where
        F: FnMut([usize; 1]) -> T,
    {
        Self {
            vals: Storage::from_fn(|idx| cb([idx])),
        }
    }

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

impl<T: Num + Copy, const N: usize> PartialEq for Vector<T, N> {
    fn eq(&self, other: &Self) -> bool {
        (0..N).all(|i| self.get([i]) == other.get([i]))
    }
}

impl<T: Num + Copy, const N: usize> Eq for Vector<T, N> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basics() {
        let a = Vector::from([1, 2, 3, 4, 5]);

        assert_eq!(a.shape(), [5]);
        assert_eq!(a.get([3]), Ok(4));
        assert_eq!(a.get([5]), Err(IndexError {}));
    }

    #[test]
    fn from_fn() {
        let a: Vector<_, 4> = Vector::from_fn(|idx| idx[0] * 2);

        assert_eq!(a, Vector::from([0, 2, 4, 6]));
    }

    #[test]
    fn dot_product() {
        let a = Vector::from([1, 2, 3]);
        let b = Vector::from([4, 5, 6]);

        assert_eq!(a.dot(&b), 32);
    }
}
