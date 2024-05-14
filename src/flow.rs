use crate::op2::{Input, Op};
use crate::tensor::Tensor;
use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub type Id = u64;

thread_local!(static NEXT_ID: RefCell<Id> = const { RefCell::new(1) });

#[derive(Debug, Clone)]
pub enum Var<Tn: Tensor> {
    Parameter(Param<Tn>),
    Output(Output<Tn>),
}

#[derive(Debug, Clone)]
pub struct Param<Tn: Tensor> {
    id: Id,
    data: Rc<RefCell<Tn>>,
    grad: Option<Rc<RefCell<Tn>>>,
}

#[derive(Debug, Clone)]
pub struct Output<Tn: Tensor> {
    id: Id,
    data: Rc<RefCell<Tn>>,
    op: Rc<dyn Op<Output = Tn>>,
    inputs: Input,
    children: HashMap<Id, Rc<dyn Any>>,
}

impl<Tn: Tensor> Var<Tn> {
    pub fn id(&self) -> Id {
        match self {
            Self::Parameter(Param { id, .. }) => *id,
            Self::Output(Output { id, .. }) => *id,
        }
    }

    fn next_id() -> Id {
        let mut id = 0;
        NEXT_ID.with(|n| {
            id = *n.borrow();
            *n.borrow_mut() = id + 1;
        });

        id
    }
}

impl<Tn: Tensor> From<Tn> for Var<Tn> {
    fn from(data: Tn) -> Self {
        Self::Parameter(Param {
            id: Self::next_id(),
            data: Rc::new(RefCell::new(data)),
            grad: None,
        })
    }
}
