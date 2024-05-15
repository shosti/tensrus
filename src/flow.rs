use crate::numeric::Numeric;
use crate::op2::{Op, OpInput, ReLU};
// use crate::op2::{Op, ReLU};
use crate::tensor::{BasicTensor, Tensor};
use std::any::Any;
use std::cell::{Ref, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

pub type Id = u64;

thread_local!(static NEXT_ID: RefCell<Id> = const { RefCell::new(1) });

#[derive(Debug, Clone)]
pub enum Var<T: Numeric> {
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
    Unary(Var<T>),
    Binary(Var<T>, Var<T>),
}

impl<T: Numeric> Var<T> {
    pub fn id(&self) -> Id {
        match self {
            Self::Parameter(p) => p.borrow().id,
            Self::Output(o) => o.borrow().id,
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

    fn new_from_unary(&self, op: Box<dyn Op<T>>) -> Self {
        let children = Children::Unary(self.clone());
        let id = Self::next_id();

        let out = match self {
            Self::Parameter(p) => {
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
            Self::Output(o) => {
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

        Self::Output(Rc::new(RefCell::new(out)))
    }

    fn backward_grad(&mut self, zero_grad: Box<dyn BasicTensor<T>>, one_grad: Box<dyn BasicTensor<T>>) {
        match self {
            Self::Parameter(p) => {
                let mut param = p.borrow_mut();
                param.grad = Some(zero_grad);
                return;
            }
            Self::Output(o) => {
                let mut topo = Vec::new();
                let mut visited = HashSet::new();

                Self::build_topo(self, &mut topo, &mut visited);
                Self::update_grads(topo, zero_grad, one_grad);
            }
        }
    }

    fn update_grads(
        topo: Vec<Var<T>>,
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

    fn children(&self) -> Vec<Var<T>> {
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

    fn build_topo(cur: &Var<T>, topo: &mut Vec<Var<T>>, visited: &mut HashSet<Id>) {
        if visited.contains(&cur.id()) {
            return;
        }

        visited.insert(cur.id());
        for child in cur.children().iter() {
            Self::build_topo(child, topo, visited);
        }
        topo.push(cur.clone());
    }

    // pub fn backward(&mut self) {
    //     match self {
    //         Self::Parameter(p) => {
    //             p.grad = Some(Rc::new(RefCell::new(Tn::zeros())));
    //             return;
    //         }
    //         Self::Output(Output { vars, id, .. }) => {
    //             let mut topo = Vec::new();
    //             let mut visited = HashSet::new();
    //             let cur = *id;
    //             let v = vars.borrow();

    //             Self::build_topo(cur, &mut topo, &mut visited, &v);

    //             for id in topo.iter().rev() {
    //                 // let var = vars.borrow()[id];
    //             }
    //         }
    //     }
    // }

    // fn build_topo(
    //     cur: Id,
    //     topo: &mut Vec<Id>,
    //     visited: &mut HashSet<Id>,
    //     vars: &Ref<HashMap<Id, VarRef>>,
    // ) {
    //     if visited.contains(&cur) {
    //         return;
    //     }

    //     visited.insert(cur);

    //     for child in &vars[&cur].children {
    //         Self::build_topo(*child, topo, visited, vars);
    //     }
    //     topo.push(cur);
    // }
}

impl<Tn: Tensor> VarOps<Tn> for Var<Tn::T> {
    fn relu(&self) -> Self {
        let op = ReLU::<Tn>::new();

        self.new_from_unary(op)
    }

    fn backward(&mut self) {
        let zero_grad = Tn::zeros();
        let one_grad = Tn::ones();
        self.backward_grad(Box::new(zero_grad), Box::new(one_grad));
    }
}

pub trait VarOps<Tn: Tensor>: Sized {
    fn backward(&mut self);
    fn relu(&self) -> Self;
}
