use crate::op2::{Op, ReLU};
use crate::tensor::Tensor;
use std::any::Any;
use std::cell::{Ref, RefCell};
use std::collections::{HashMap, HashSet};
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
    vars: Rc<RefCell<HashMap<Id, VarRef>>>,
}

#[derive(Debug)]
struct VarRef {
    var: Box<dyn Any>, // Box<Var<??>>
    children: Vec<Id>,
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

    fn as_ref(&self) -> VarRef {
        match self {
            Self::Parameter(_) => VarRef {
                var: Box::new(self.clone()),
                children: vec![],
            },
            Self::Output(o) => {
                let children = match o.children {
                    Children::Unary(id) => vec![id],
                    Children::Binary(a, b) => vec![a, b],
                };
                VarRef {
                    var: Box::new(self.clone()),
                    children,
                }
            }
        }
    }

    fn new_from_unary(&self, op: Rc<dyn Op<Output = Tn>>) -> Self {
        let id = Self::next_id();
        let vars = match self {
            Self::Parameter(_) => {
                let mut h: HashMap<Id, VarRef> = HashMap::new();
                h.insert(self.id(), self.as_ref());
                Rc::new(RefCell::new(h))
            }
            Self::Output(Output { vars, .. }) => {
                let mut h = vars.borrow_mut();
                h.insert(self.id(), self.as_ref());
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

    pub fn backward(&mut self) {
        match self {
            Self::Parameter(p) => {
                p.grad = Some(Rc::new(RefCell::new(Tn::zeros())));
                return;
            }
            Self::Output(Output { vars, id, .. }) => {
                let mut topo = Vec::new();
                let mut visited = HashSet::new();
                let cur = *id;
                let v = vars.borrow();

                Self::build_topo(cur, &mut topo, &mut visited, &v);
            }
        }
    }

    fn build_topo(
        cur: Id,
        topo: &mut Vec<Id>,
        visited: &mut HashSet<Id>,
        vars: &Ref<HashMap<Id, VarRef>>,
    ) {
        if visited.contains(&cur) {
            return;
        }

        visited.insert(cur);

        for child in &vars[&cur].children {
            Self::build_topo(*child, topo, visited, vars);
        }
        topo.push(cur);
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
