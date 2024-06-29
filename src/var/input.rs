use super::Id;
use crate::{
    differentiable::{Differentiable, DifferentiableTensor},
    dyn_tensor::DynTensor,
};
use std::{
    cell::{Ref, RefCell},
    marker::PhantomData,
    rc::Rc,
};

#[derive(Debug, Clone)]
pub struct Input<Tn> {
    pub(super) inner: Rc<RefCell<InputInner>>,
    marker: PhantomData<Tn>,
}

#[derive(Debug)]
pub(super) struct InputInner {
    pub(super) id: Id,
    pub(super) data: Box<dyn DynTensor>,
}

impl<Tn> Input<Tn>
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

    pub fn into_data(self) -> Option<Tn> {
        let inner = Rc::into_inner(self.inner)?.into_inner();
        Some(*Tn::from_dyn(inner.data))
    }

    pub(super) fn new(id: Id, data: Tn) -> Self {
        let inner = InputInner {
            id,
            data: Box::new(data),
        };

        Self {
            inner: Rc::new(RefCell::new(inner)),
            marker: PhantomData,
        }
    }
}
