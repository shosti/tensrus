use crate::tensor::{IndexError, Tensor};
use num::Num;
use std::cell::RefCell;
use std::ops::MulAssign;
use std::rc::Rc;

#[derive(Debug)]
pub struct Vector<T: Num + Copy, const N: usize> {
    vals: Rc<RefCell<Vec<T>>>,
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
            vals: Rc::new(RefCell::new(Vec::from(vals))),
        }
    }
}

impl<T: Num + Copy, const N: usize> Tensor<T, 1> for Vector<T, N> {
    type Transpose = Vector<T, N>;

    fn from_fn<F>(mut cb: F) -> Self
    where
        F: FnMut([usize; 1]) -> T,
    {
        let vals = Rc::new(RefCell::new((0..N).map(|idx| cb([idx])).collect()));
        Self { vals }
    }

    fn shape(&self) -> [usize; 1] {
        [N]
    }

    fn get(&self, idx: [usize; 1]) -> Result<T, IndexError> {
        let [i] = idx;
        if i >= N {
            return Err(IndexError {});
        }

        Ok(self.vals.borrow()[i])
    }

    fn set(&mut self, idx: [usize; 1], val: T) -> Result<(), IndexError> {
        let [i] = idx;
        if i >= N {
            return Err(IndexError {});
        }
        self.vals.borrow_mut()[i] = val;

        Ok(())
    }

    fn transpose(&self) -> Self::Transpose {
        Self::Transpose {
            vals: self.vals.clone(),
        }
    }
}

impl<T: Num + Copy, const N: usize> MulAssign<T> for Vector<T, N> {
    fn mul_assign(&mut self, other: T) {
        self.vals
            .borrow_mut()
            .iter_mut()
            .for_each(|n| *n = *n * other);
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
