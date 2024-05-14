use crate::op2::{Input, Op};
use crate::tensor::Tensor;
use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

type Id = u64;

thread_local!(static NEXT_ID: RefCell<Id> = const { RefCell::new(1) });

#[derive(Debug, Clone)]
pub struct Flow {
    var_id: Id,
    var: Rc<dyn Any>, // Is actually Rc<RefCell<Var<Tn>>>
}

#[derive(Debug)]
pub enum Var<Tn: Tensor> {
    Parameter(Param<Tn>),
    Output(Output<Tn>),
}

#[derive(Debug)]
pub struct Param<Tn: Tensor> {
    id: Id,
    data: Tn,
    grad: Option<Tn>,
}

#[derive(Debug)]
pub struct Output<Tn: Tensor> {
    id: Id,
    data: Tn,
    op: Box<dyn Op<Output = Tn>>,
    inputs: Input,
}

impl Flow {
    pub fn into_var<Tn: Tensor>(self) -> Rc<RefCell<Var<Tn>>> {
        let out: Rc<RefCell<Var<Tn>>> = self.var.downcast().unwrap();
        out
    }
}

impl<Tn: Tensor> Var<Tn> {
    pub fn param(data: Tn) -> Self {
        Self::Parameter(Param {
            id: Self::next_id(),
            data,
            grad: None,
        })
    }

    pub fn map(&self, f: impl FnOnce(&Tn) -> Tn) -> Tn {
        match self {
            Self::Parameter(Param { data, .. }) => f(data),
            Self::Output(Output { data, .. }) => f(data),
        }
    }

    pub fn output(op: Box<dyn Op<Output = Tn>>, inputs: Input) -> Self {
        let data = op.forward(&inputs);
        Self::Output(Output {
            id: Self::next_id(),
            data,
            op,
            inputs,
        })
    }

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

impl<Tn: Tensor> From<Var<Tn>> for Flow {
    fn from(v: Var<Tn>) -> Self {
        Self {
            var_id: v.id(),
            var: Rc::new(RefCell::new(v))
        }
    }
}
