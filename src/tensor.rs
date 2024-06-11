use std::fmt::Debug;
use std::{
    any::Any,
    iter::Map,
    ops::{Index, Mul},
};

use num::{One, Zero};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::numeric::Numeric;
use crate::scalar::Scalar;
use crate::shape::{Shape, Shaped};
use crate::storage::TensorStorage;
use crate::view::View;

pub trait BasicTensor<T: Numeric>: Debug + for<'a> Index<&'a [usize], Output = T> {
    fn as_any(&self) -> &dyn Any;
    fn as_any_boxed(self: Box<Self>) -> Box<dyn Any>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn clone_boxed(&self) -> Box<dyn BasicTensor<T>>;

    // Returns a new tensor of zeros with the same shape as self
    fn zeros_with_shape(&self) -> Box<dyn BasicTensor<T>>;

    // Returns a new tensor of ones with the same shape as self
    fn ones_with_shape(&self) -> Box<dyn BasicTensor<T>>;

    fn add(self: Box<Self>, other: &dyn BasicTensor<T>, scale: T) -> Box<dyn BasicTensor<T>>;
}

pub trait TensorIndex: AsRef<[usize]> + Copy {
    fn from_slice(s: &[usize]) -> Self;
}

impl<const R: usize> TensorIndex for [usize; R] {
    fn from_slice(s: &[usize]) -> Self {
        let mut ret = [0; R];
        ret.copy_from_slice(&s[..R]);
        ret
    }
}

pub trait TensorLike: Shaped + TensorStorage<Self::T> + Indexable {}

pub trait Indexable: for<'a> Index<&'a Self::Idx, Output = Self::T> {
    type Idx: TensorIndex;
    type T: Numeric;

    fn num_elems() -> usize;
    fn default_idx() -> Self::Idx;
    fn next_idx(&self, idx: &Self::Idx) -> Option<Self::Idx>;
}

pub trait Tensor:
    Clone
    + BasicTensor<Self::T>
    + Mul<Self::T, Output = Self>
    + TensorLike
    + Eq
    + FromIterator<Self::T>
    + 'static
{
    // Required methods
    fn rank() -> usize;
    fn shape() -> Shape;

    fn from_fn(f: impl Fn(&Self::Idx) -> Self::T) -> Self;
    fn set(self, idx: &Self::Idx, f: impl Fn(Self::T) -> Self::T) -> Self;

    // Provided methods
    fn map(mut self, f: impl Fn(&Self::Idx, Self::T) -> Self::T) -> Self {
        let mut next_idx = Some(Self::default_idx());
        while let Some(idx) = next_idx {
            self = self.set(&idx, |val| f(&idx, val));
            next_idx = self.next_idx(&idx);
        }

        self
    }

    fn view(&self) -> View<Self>
    where
        Self: TensorStorage<Self::T>,
    {
        self.into()
    }
    fn repeat(n: Self::T) -> Self {
        Self::from_fn(|_| n)
    }
    fn zeros() -> Self {
        Self::repeat(Self::T::zero())
    }
    fn ones() -> Self {
        Self::repeat(Self::T::one())
    }
    fn rand(d: impl Distribution<Self::T>, rng: &mut impl Rng) -> Self {
        d.sample_iter(rng)
            .take(<Self as Indexable>::num_elems())
            .collect()
    }
    fn randn(rng: &mut impl Rng) -> Self
    where
        StandardNormal: Distribution<Self::T>,
    {
        Self::rand(StandardNormal, rng)
    }

    fn iter(&self) -> TensorIterator<Self> {
        TensorIterator::new(self)
    }

    fn ref_from_basic(from: &dyn BasicTensor<Self::T>) -> &Self {
        let any_ref = from.as_any();
        any_ref.downcast_ref().unwrap()
    }

    fn from_basic(from: &dyn BasicTensor<Self::T>) -> Self {
        if from.len() != <Self as Indexable>::num_elems() {
            panic!("cannot create a Tensor from a BasicTensor unless the number of elements is identical");
        }

        Self::from_fn(|idx| from[idx.as_ref()])
    }

    fn from_basic_boxed(from: Box<dyn BasicTensor<Self::T>>) -> Box<Self> {
        from.as_any_boxed().downcast().unwrap()
    }

    fn sum(&self) -> Scalar<Self::T> {
        let s: Self::T = self.iter().values().sum();
        Scalar::from(s)
    }

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

pub struct TensorIterator<'a, Tn>
where
    Tn: Indexable,
{
    t: &'a Tn,
    cur: Option<Tn::Idx>,
}

impl<'a, Tn> TensorIterator<'a, Tn>
where
    Tn: Indexable,
{
    pub fn new(t: &'a Tn) -> Self {
        Self {
            t,
            cur: Some(Tn::default_idx()),
        }
    }

    pub fn values(self) -> Map<TensorIterator<'a, Tn>, impl FnMut((Tn::Idx, Tn::T)) -> Tn::T> {
        self.map(|(_, v)| v)
    }
}

impl<'a, Tn> Iterator for TensorIterator<'a, Tn>
where
    Tn: Indexable,
{
    type Item = (Tn::Idx, Tn::T);

    fn next(&mut self) -> Option<Self::Item> {
        match &self.cur {
            None => None,
            Some(idx) => {
                let cur_idx = *idx;
                let item = (cur_idx, self.t[&cur_idx]);
                self.cur = self.t.next_idx(&cur_idx);

                Some(item)
            }
        }
    }
}
