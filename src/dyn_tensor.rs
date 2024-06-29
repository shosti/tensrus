use crate::tensor::Tensor;
use num::{One, Zero};
use static_assertions::assert_obj_safe;
use std::{any::Any, fmt::Debug};

pub trait DynTensor: Debug {
    fn as_any(&self) -> &dyn Any;
    fn as_any_boxed(self: Box<Self>) -> Box<dyn Any>;

    fn len(&self) -> usize;
    fn ones_with_shape(&self) -> Box<dyn DynTensor>;
    fn zeros_with_shape(&self) -> Box<dyn DynTensor>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

assert_obj_safe!(DynTensor);

impl<Tn> DynTensor for Tn
where
    Tn: Tensor + Debug + 'static,
    Tn::T: Copy + One + Zero,
{
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_boxed(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    fn len(&self) -> usize {
        Self::num_elems()
    }

    fn ones_with_shape(&self) -> Box<dyn DynTensor> {
        Box::new(Self::ones())
    }
    fn zeros_with_shape(&self) -> Box<dyn DynTensor> {
        Box::new(Self::zeros())
    }
}

pub trait FromDynTensor: 'static {
    fn ref_from_dyn(from: &dyn DynTensor) -> &Self;
    fn from_dyn(from: Box<dyn DynTensor>) -> Box<Self>;
}

impl<Tn> FromDynTensor for Tn
where
    Tn: Tensor + 'static,
{
    fn ref_from_dyn(from: &dyn DynTensor) -> &Self {
        let any_ref = from.as_any();
        any_ref.downcast_ref().unwrap()
    }

    fn from_dyn(from: Box<dyn DynTensor>) -> Box<Self> {
        from.as_any_boxed().downcast().unwrap()
    }
}
