use crate::differentiable::{Differentiable, DifferentiableTensor};
use crate::dyn_tensor::DynTensor;
use crate::errors::UpdateFromGradError;
use std::{
    cell::{Ref, RefCell},
    marker::PhantomData,
    rc::Rc,
};

use super::Id;

#[derive(Debug, Clone)]
pub struct Param<Tn> {
    pub(super) inner: Rc<RefCell<ParamInner>>,
    marker: PhantomData<Tn>,
}

#[derive(Debug)]
pub(super) struct ParamInner {
    pub(super) id: Id,
    pub(super) data: Box<dyn DynTensor>,
    grad: Option<Box<dyn DynTensor>>,
}

impl<Tn> Param<Tn>
where
    Tn: DifferentiableTensor,
    Tn::T: Differentiable,
{
    pub fn id(&self) -> Id {
        self.inner.borrow().id
    }

    pub fn data(&self) -> Ref<Tn> {
        Ref::map(self.inner.borrow(), |inner| {
            Tn::ref_from_dyn(inner.data.as_ref())
        })
    }

    pub fn grad(&self) -> Option<Ref<Tn>> {
        self.inner.borrow().grad.as_ref()?;

        Some(Ref::map(self.inner.borrow(), |inner| {
            Tn::ref_from_dyn(inner.grad.as_ref().unwrap().as_ref())
        }))
    }

    pub(super) fn init_grad(&self) {
        let mut inner = self.inner.borrow_mut();
        inner.grad = Some(inner.data.ones_with_shape());
    }

    pub fn update_from_grad(&self, epsilon: Tn::T) -> Result<(), UpdateFromGradError> {
        let mut inner = self.inner.borrow_mut();
        let grad_dyn = inner
            .grad
            .take()
            .ok_or(UpdateFromGradError::GradNotCalculated)?;
        let grad = *Tn::from_dyn(grad_dyn) * epsilon;
        inner.data = Box::new(grad + Tn::ref_from_dyn(inner.data.as_ref()));

        Ok(())
    }

    pub fn into_data(self) -> Option<Tn> {
        let inner = Rc::into_inner(self.inner)?.into_inner();
        Some(*Tn::from_dyn(inner.data))
    }

    pub(super) fn new(id: Id, data: Tn) -> Self {
        let inner = ParamInner {
            id,
            data: Box::new(data),
            grad: None,
        };

        Self {
            inner: Rc::new(RefCell::new(inner)),
            marker: PhantomData,
        }
    }
}

impl ParamInner {
    pub(super) fn set_grad(&mut self, grad: Box<dyn DynTensor>) {
        self.grad = Some(grad);
    }
}
