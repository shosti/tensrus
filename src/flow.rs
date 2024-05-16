use crate::numeric::Numeric;
use crate::op2::{Op, OpInput, ReLU};
use crate::tensor::{BasicTensor, Tensor};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::rc::Rc;

pub type Id = u64;

thread_local!(static NEXT_ID: RefCell<Id> = const { RefCell::new(1) });

#[derive(Debug, Clone)]
pub enum Var<Tn: Tensor> {
    Parameter(Rc<RefCell<Param<Tn::T>>>, PhantomData<Tn>),
    Output(Rc<RefCell<Output<Tn::T>>>, PhantomData<Tn>),
}

#[derive(Debug, Clone)]
enum VarRef<T: Numeric> {
    Parameter(Rc<RefCell<Param<T>>>),
    Output(Rc<RefCell<Output<T>>>),
}

#[derive(Debug)]
pub struct Param<T: Numeric> {
    id: Id,
    data: Box<dyn BasicTensor<T>>,
    grad: Option<Box<dyn BasicTensor<T>>>,
}

#[derive(Debug)]
pub struct Output<T: Numeric> {
    id: Id,
    data: Box<dyn BasicTensor<T>>,
    op: Box<dyn Op<T>>,
    children: Children<T>,
}

#[derive(Debug)]
enum Children<T: Numeric> {
    Unary(VarRef<T>),
    Binary(VarRef<T>, VarRef<T>),
}

impl<Tn: Tensor> Var<Tn> {
    fn next_id() -> Id {
        let mut id = 0;
        NEXT_ID.with(|n| {
            id = *n.borrow();
            *n.borrow_mut() = id + 1;
        });

        id
    }

    fn new_from_unary(&self, op: Box<dyn Op<Tn::T>>) -> Self {
        let children = Children::Unary(self.into());
        let id = Self::next_id();

        let out = match self {
            Self::Parameter(p, _) => {
                let param = p.borrow();
                let input = OpInput::Unary(&param.data);
                let data = op.forward(input);
                Output {
                    id,
                    data,
                    op,
                    children,
                }
            }
            Self::Output(o, _) => {
                let last_out = o.borrow();
                let input = OpInput::Unary(&last_out.data);
                let data = op.forward(input);
                Output {
                    id,
                    data,
                    op,
                    children,
                }
            }
        };

        Self::Output(Rc::new(RefCell::new(out)), PhantomData)
    }
}

impl<T: Numeric> VarRef<T> {
    fn id(&self) -> Id {
        match self {
            Self::Parameter(p) => p.borrow().id,
            Self::Output(o) => o.borrow().id,
        }
    }

    fn backward_grad(
        &mut self,
        zero_grad: Box<dyn BasicTensor<T>>,
        one_grad: Box<dyn BasicTensor<T>>,
    ) {
        match self {
            Self::Parameter(p) => {
                let mut param = p.borrow_mut();
                param.grad = Some(zero_grad.clone());
                return;
            }
            Self::Output(_) => {
                let mut topo = Vec::new();
                let mut visited = HashSet::new();

                Self::build_topo(self, &mut topo, &mut visited);
                Self::update_grads(topo, zero_grad, one_grad);
            }
        }
    }

    fn update_grads(
        topo: Vec<VarRef<T>>,
        zero_grad: Box<dyn BasicTensor<T>>,
        one_grad: Box<dyn BasicTensor<T>>,
    ) {
        let mut accumulators = HashMap::<Id, Box<dyn BasicTensor<T>>>::new();
        for v in topo.iter().rev() {
            match v {
                Self::Parameter(p) => {
                    todo!()
                }
                Self::Output(p) => {
                    todo!()
                }
            }
        }
    }

    fn children(&self) -> Vec<VarRef<T>> {
        match self {
            Self::Parameter(_) => vec![],
            Self::Output(o) => {
                let out = o.borrow();
                match &out.children {
                    Children::Unary(c) => vec![c.clone()],
                    Children::Binary(c1, c2) => vec![c1.clone(), c2.clone()],
                }
            }
        }
    }

    fn build_topo(cur: &VarRef<T>, topo: &mut Vec<VarRef<T>>, visited: &mut HashSet<Id>) {
        if visited.contains(&cur.id()) {
            return;
        }

        visited.insert(cur.id());
        for child in cur.children().iter() {
            Self::build_topo(child, topo, visited);
        }
        topo.push(cur.clone());
    }
}

impl<Tn: Tensor> From<&Var<Tn>> for VarRef<Tn::T> {
    fn from(v: &Var<Tn>) -> Self {
        match v {
            Var::Parameter(p, _) => Self::Parameter(p.clone()),
            Var::Output(o, _) => Self::Output(o.clone()),
        }
    }
}

impl<Tn: Tensor> Var<Tn> {
    pub fn relu(&self) -> Self {
        let op = ReLU::<Tn>::new();

        self.new_from_unary(op)
    }

    pub fn backward(&self) {
        let zero_grad = Tn::zeros();
        let one_grad = Tn::ones();
        VarRef::from(self).backward_grad(Box::new(zero_grad), Box::new(one_grad));
    }
}

impl<Tn: Tensor> From<Tn> for Var<Tn> {
    fn from(t: Tn) -> Self {
        let param = Param {
            id: Self::next_id(),
            data: Box::new(t),
            grad: None,
        };

        Self::Parameter(Rc::new(RefCell::new(param)), PhantomData)
    }
}
