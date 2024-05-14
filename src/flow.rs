use crate::op2::{Op, ReLU};
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
    children: Children,
    vars: Rc<RefCell<HashMap<Id, Box<dyn Any>>>>, // HashMap Any value is Var
}

#[derive(Debug, Clone)]
enum Children {
    Unary(Id),
    Binary(Id, Id),
}

impl<Tn: Tensor> Var<Tn> {
    pub fn id(&self) -> Id {
        match self {
            Self::Parameter(Param { id, .. }) => *id,
            Self::Output(Output { id, .. }) => *id,
        }
    }

    pub fn data(&self) -> Rc<RefCell<Tn>> {
        match self {
            Self::Parameter(Param { data, .. }) => data.clone(),
            Self::Output(Output { data, .. }) => data.clone(),
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

    fn new_from_unary(&self, op: Rc<dyn Op<Output = Tn>>) -> Self {
        let id = Self::next_id();
        let vars = match self {
            Self::Parameter(_) => {
                let mut h: HashMap<Id, Box<dyn Any>> = HashMap::new();
                h.insert(self.id(), Box::new(self.clone()));
                Rc::new(RefCell::new(h))
            }
            Self::Output(Output { vars, .. }) => {
                let mut h = vars.borrow_mut();
                h.insert(self.id(), Box::new(self.clone()));
                vars.clone()
            }
        };

        let data = op.forward(self.data().into());
        Self::Output(Output {
            id,
            children: Children::Unary(self.id()),
            op,
            vars,
            data: Rc::new(RefCell::new(data)),
        })
    }

    pub fn relu(&self) -> Self {
        let op = Rc::new(ReLU::new());

        self.new_from_unary(op)
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
