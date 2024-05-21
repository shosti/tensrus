use crate::{
    numeric::Numeric,
    op::{AddOp, MulOp, NoOp, Op, PowOp, ReluOp},
    render::{Graphable, Node, NodeId},
    scalar::Scalar,
    tensor::Tensor,
};

use std::{
    any::Any,
    cell::RefCell,
    cmp::Ordering,
    collections::HashSet,
    fmt::Debug,
    hash::{Hash, Hasher},
    iter::Sum,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

thread_local!(static NEXT_ID: RefCell<u64> = const { RefCell::new(1) });

#[derive(Debug, Clone)]
pub struct Var<Tn: Tensor> {
    pub id: u64,
    pub data: Tn,
    pub grad: Option<Tn>,
    op: Rc<RefCell<dyn Op>>,
}

#[derive(Debug, Clone)]
pub struct VarRef {
    pub id: u64,
    data: Rc<dyn Any>,
    grad: Rc<dyn Any>,
    op: Rc<RefCell<dyn Op>>,
}

impl<Tn: Tensor> Var<Tn> {
    pub fn new(data: Tn) -> Self {
        Self {
            id: Self::next_id(),
            data,
            grad: None,
            op: Rc::new(RefCell::new(NoOp {})),
        }
    }

    pub fn new_from_op(data: Tn, op: impl Op) -> Self {
        Self {
            id: Self::next_id(),
            data,
            grad: None,
            op: Rc::new(RefCell::new(op)),
        }
    }

    fn next_id() -> u64 {
        let mut id = 0;
        NEXT_ID.with(|n| {
            id = *n.borrow();
            *n.borrow_mut() = id + 1;
        });

        id
    }

    pub fn relu(&self) -> Self {
        ReluOp::create_flow(self.clone())
    }

    pub fn update_from_grad(&mut self, epsilon: Tn::T) {
        if let Some(grad) = self.grad.take() {
            self.data = (grad * -epsilon) + &self.data;
        }
    }

    pub fn update_grad(&mut self, _f: impl Fn(Tn::T, Tn::T) -> Tn::T) {
        // match self.grad.ta
    }

    // TODO: We don't need this I think?
    pub fn zero_grad(&mut self) {
        self.grad = None;
    }

    fn build_trace(
        val: &VarRef,
        nodes: &mut HashSet<VarRef>,
        edges: &mut HashSet<(VarRef, VarRef)>,
    ) {
        if !nodes.contains(val) {
            nodes.insert(val.clone());
            for child in val.op.borrow().children().iter() {
                edges.insert((child.clone(), val.clone()));
                Self::build_trace(child, nodes, edges);
            }
        }
    }
}

impl<T: Numeric> Graphable for Var<Scalar<T>> {
    fn trace(&self) -> (HashSet<Node>, HashSet<(NodeId, NodeId)>) {
        let mut nodes = HashSet::new();
        let mut edges = HashSet::new();
        let val: VarRef = self.clone().into();

        Self::build_trace(&val, &mut nodes, &mut edges);

        let g_nodes = nodes
            .iter()
            .map(|n| {
                let data: &Scalar<T> = n.data();
                let grad: &Scalar<T> = n.grad();
                let label = format!(
                    "{} | data: {:?} | grad: {:?} | {:?}",
                    n.id,
                    data,
                    grad,
                    n.op()
                );
                let id = format!("{}", n.id);

                Node { id, label }
            })
            .collect();
        let g_edges = edges
            .iter()
            .map(|(from, to)| (format!("{}", from.id), format!("{}", to.id)))
            .collect();

        (g_nodes, g_edges)
    }
}

impl<T: Numeric> From<T> for Var<Scalar<T>> {
    fn from(val: T) -> Self {
        Var::new(Scalar::from(val))
    }
}

impl<T: Numeric> Var<Scalar<T>> {
    pub fn backward(&mut self) {
        // let mut topo = Vec::new();
        // let mut visited = HashSet::new();
        // let cur: VarRef = self.clone().into();

        // Self::build_topo(&cur, &mut topo, &mut visited);

        // self.grad.update(|_| T::one());
        // for flow in topo.iter().rev() {
        //     flow.op.borrow_mut().backward(flow);
        // }
    }

    // fn build_topo(cur: &VarRef, topo: &mut Vec<VarRef>, visited: &mut HashSet<u64>) {
    //     if visited.contains(&cur.id) {
    //         return;
    //     }

    //     visited.insert(cur.id);
    //     for child in cur.op.borrow().children().iter() {
    //         Self::build_topo(child, topo, visited);
    //     }
    //     topo.push(cur.clone());
    // }

    pub fn pow(&self, n: T) -> Self {
        PowOp::create_flow(self.clone(), n)
    }
}

impl<Tn: Tensor> Add for Var<Tn> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        AddOp::create_flow(&self, &other)
    }
}

impl<T: Numeric> Mul for Var<Scalar<T>> {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        MulOp::create_flow(self, other)
    }
}

impl<T: Numeric> Div for Var<Scalar<T>> {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        let inv = other.pow(-T::one());

        self * inv
    }
}

impl<T: Numeric> Sub for Var<Scalar<T>> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self + (-other)
    }
}

impl<T: Numeric> Neg for Var<Scalar<T>> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * Var::new(Scalar::from(-T::one()))
    }
}

impl<Tn: Tensor> PartialEq for Var<Tn> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<Tn: Tensor> Eq for Var<Tn> {}

impl VarRef {
    pub fn data<Tn: Tensor>(&self) -> &Tn {
        let data: &Tn = self.data.downcast_ref().unwrap();
        data
    }

    pub fn grad<Tn: Tensor>(&self) -> &Tn {
        let grad: &Tn = self.grad.downcast_ref().unwrap();
        grad
    }

    pub fn op(&self) -> String {
        format!("{:?}", self.op.borrow())
    }
}

impl Hash for VarRef {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.id.hash(state)
    }
}

impl PartialOrd for VarRef {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.id.cmp(&other.id))
    }
}

impl Ord for VarRef {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl PartialEq for VarRef {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for VarRef {}

impl<T: Numeric> Sum for Var<Scalar<T>> {
    fn sum<I: Iterator<Item = Var<Scalar<T>>>>(mut _iter: I) -> Self {
        todo!()
        // let first = iter.next();
        // if first.is_none() {
        //     return Var::from(T::zero());
        // }
        // let mut res = first.unwrap();

        // for next in iter {
        //     res = res + next;
        // }

        // res
    }
}

impl<Tn: Tensor> From<Var<Tn>> for VarRef {
    fn from(flow: Var<Tn>) -> Self {
        Self {
            id: flow.id,
            data: Rc::new(flow.data.clone()),
            grad: Rc::new(flow.grad.clone()),
            op: flow.op.clone(),
        }
    }
}
