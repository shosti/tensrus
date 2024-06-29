use crate::{
    differentiable::{Differentiable, DifferentiableTensor},
    generic_tensor::IntoGeneric,
    shape::{self, Broadcastable, Shape, Shaped},
    storage::Storage,
    tensor::{Indexable, Tensor},
};
use std::ops::Index;

pub const RANK: usize = 0;

#[derive(TensorStorage, OwnedTensorStorage, Tensor, Debug, Clone)]
pub struct Scalar<T> {
    storage: Storage<T>,
}

impl<T> Scalar<T> {
    pub const fn shape() -> Shape {
        shape::rank0()
    }

    pub fn val(&self) -> T
    where
        T: Clone,
    {
        self[&[]].clone()
    }
}

impl<T> Shaped for Scalar<T> {
    fn rank() -> usize {
        RANK
    }

    fn shape() -> Shape {
        Self::shape()
    }
}

impl<T> From<T> for Scalar<T> {
    fn from(val: T) -> Self {
        Self {
            storage: [val].into(),
        }
    }
}

impl<T> IntoGeneric<T, { RANK }, { Self::shape() }> for Scalar<T> {}

impl<T> Index<&[usize; RANK]> for Scalar<T> {
    type Output = T;

    fn index(&self, idx: &[usize; RANK]) -> &Self::Output {
        self.storage
            .index(idx, Self::rank(), Self::shape())
            .unwrap()
    }
}

impl<T> Index<&[usize; RANK]> for &Scalar<T> {
    type Output = T;

    fn index(&self, idx: &[usize; RANK]) -> &Self::Output {
        (*self).index(idx)
    }
}

impl<T> Indexable for Scalar<T> {
    type Idx = [usize; RANK];
    type T = T;
}

impl<T> DifferentiableTensor for Scalar<T> where T: Differentiable {}

impl<Tn: Tensor> Broadcastable<Tn> for Scalar<Tn::T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::Vector;

    #[test]
    fn basics() {
        let a = Scalar::from(42);

        assert_eq!(a[&[]], 42);
        assert_eq!(a.val(), 42);
    }

    #[test]
    fn multiplication() {
        assert_eq!(Scalar::from(6) * Scalar::from(7), Scalar::from(42));

        let x = Vector::from([1, 2, 3]);
        let a = Scalar::from(6);

        assert_eq!(x * a, Vector::from([6, 12, 18]));
    }
}
