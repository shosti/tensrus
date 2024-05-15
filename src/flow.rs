use crate::numeric::Numeric;
use crate::op2::ReLU;
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
    // op: Box<dyn Op<Output = Tn>>,
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

    // fn new_from_unary(&self, op: Rc<dyn Op<Output = Tn>>) -> Self {
    //     let id = Self::next_id();
    //     let vars = match self {
    //         Self::Parameter(_) => {
    //             let mut h: HashMap<Id, VarRef> = HashMap::new();
    //             h.insert(self.id(), self.as_ref());
    //             Rc::new(RefCell::new(h))
    //         }
    //         Self::Output(Output { vars, .. }) => {
    //             let mut h = vars.borrow_mut();
    //             h.insert(self.id(), self.as_ref());
    //             vars.clone()
    //         }
    //     };

    //     let data = op.forward(self.data().into());
    //     Self::Output(Output {
    //         id,
    //         children: Children::Unary(self.id()),
    //         op,
    //         vars,
    //         data: Rc::new(RefCell::new(data)),
    //     })
    // }

    // pub fn relu(&self) -> Self {
    //     let op = Rc::new(ReLU::new());

    //     self.new_from_unary(op)
    // }

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

impl<Tn: Tensor> VarOps<Tn> for Var<Tn::T> {}

pub trait VarOps<Tn: Tensor>: Sized {
    fn relu(&self) -> Self {
        let op = ReLU::<Tn>::new();

        todo!()
        // self.new_from_unary(op)
    }
}
