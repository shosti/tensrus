use crate::numeric::Numeric;
use crate::op2::{BackwardOutput, ForwardInput, Op, ReLU};
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
                let input = ForwardInput::Unary(&param.data);
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
                let input = ForwardInput::Unary(&last_out.data);
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

    fn backward(&mut self) {
        match self {
            Self::Parameter(p) => {
                let mut param = p.borrow_mut();
                param.grad = Some(param.data.ones_with_shape());
                return;
            }
            Self::Output(_) => {
                let mut topo = Vec::new();
                let mut visited = HashSet::new();

                Self::build_topo(self, &mut topo, &mut visited);
                self.update_grads(topo);
            }
        }
    }

    fn update_grads(&self, topo: Vec<VarRef<T>>) {
        let mut accumulators = HashMap::<Id, Box<dyn BasicTensor<T>>>::new();
        let ones = match self {
            Self::Parameter(p) => p.borrow().data.ones_with_shape(),
            Self::Output(o) => o.borrow().data.ones_with_shape(),
        };
        accumulators.insert(self.id(), ones);

        for v in topo.iter().rev() {
            match v {
                Self::Parameter(_) => {
                    // We're at a parameter which has no children; nothing left
                    // to do
                }
                Self::Output(o) => {
                    let out = o.borrow();
                    let out_grad = accumulators
                        .entry(out.id)
                        .or_insert_with(|| out.data.ones_with_shape());
                    let grads = out.op.backward(&out.data, out_grad);
                    match &out.children {
                        Children::Unary(c) => {
                            if let BackwardOutput::Unary(grad) = grads {
                                Self::update_grad(c, grad, &mut accumulators);
                            } else {
                                panic!(
                                    "op backwards() outputted non-unary grads for unary children"
                                );
                            }
                        }
                        Children::Binary(c1, c2) => {
                            if let BackwardOutput::Binary(g1, g2) = grads {
                                Self::update_grad(c1, g1, &mut accumulators);
                                Self::update_grad(c2, g2, &mut accumulators);
                            } else {
                                panic!(
                                    "op backwards() outputted non-binary grads for binary children"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    fn update_grad(
        var: &VarRef<T>,
        grad: Box<dyn BasicTensor<T>>,
        accumulators: &mut HashMap<Id, Box<dyn BasicTensor<T>>>,
    ) {
        let k = var.id();
        match accumulators.remove(&k) {
            Some(old_grad) => {
                accumulators.insert(k, old_grad.add(&grad));
            }
            None => {
                accumulators.insert(k, grad);
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
        VarRef::from(self).backward();
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
