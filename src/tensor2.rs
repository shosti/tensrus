use std::ops::Index;

use num::{One, Zero};
// use rand::Rng;
// use rand_distr::{Distribution, StandardNormal};

use crate::numeric::Numeric;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    Normal,
    Transposed,
}

pub trait Tensor2:
    Clone
    // + for<'a> Add<&'a Self, Output = Self>
    // + Mul<Self::T, Output = Self>
    + for<'a> Index<&'a Self::Idx, Output = Self::T>
    // + PartialEq
    // + FromIterator<Self::T>
    + 'static
{
    type T: Numeric;
    type Idx: AsRef<[usize]> + Copy + 'static;

    // Required methods
    fn from_fn(f: impl Fn(&Self::Idx) -> Self::T) -> Self;
    fn map(self, f: impl Fn(&Self::Idx, Self::T) -> Self::T) -> Self;

    fn num_elems() -> usize;
    fn default_idx() -> Self::Idx;
    fn next_idx(&self, idx: &Self::Idx) -> Option<Self::Idx>;

    // Provided methods
    fn repeat(n: Self::T) -> Self {
        Self::from_fn(|_| n)
    }
    fn zeros() -> Self {
        Self::repeat(Self::T::zero())
    }
    fn ones() -> Self {
        Self::repeat(Self::T::one())
    }
    // fn rand(d: impl Distribution<Self::T>, rng: &mut impl Rng) -> Self {
    //     d.sample_iter(rng)
    //         .take(<Self as Tensor2>::num_elems())
    //         .collect()
    // }
    // fn randn(rng: &mut impl Rng) -> Self
    // where
    //     Self: FromIterator<Self::T>,
    //     StandardNormal: Distribution<Self::T>,
    // {
    //     Self::rand(StandardNormal, rng)
    // }

    // fn iter(&self) -> TensorIterator<Self> {
    //     TensorIterator::new(self)
    // }

    // fn ref_from_basic(from: &dyn BasicTensor<Self::T>) -> &Self {
    //     let any_ref = from.as_any();
    //     any_ref.downcast_ref().unwrap()
    // }

    // fn from_basic(from: &dyn BasicTensor<Self::T>) -> Self {
    //     if from.num_elems() != <Self as Tensor>::num_elems() {
    //         panic!("cannot create a Tensor from a BasicTensor unless the number of elements is identical");
    //     }

    //     Self::from_fn(|idx| from[idx.as_ref()])
    // }

    // fn from_basic_boxed(from: Box<dyn BasicTensor<Self::T>>) -> Box<Self> {
    //     from.as_any_boxed().downcast().unwrap()
    // }

    // fn sum(&self) -> Scalar<Self::T> {
    //     let s: Self::T = self.iter().values().sum();
    //     Scalar::from(s)
    // }

    // Operations
    fn relu(self) -> Self {
        self.map(|_, val| {
            if val < Self::T::zero() {
                Self::T::zero()
            } else {
                val
            }
        })
    }
}
