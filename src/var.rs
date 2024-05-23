use crate::numeric::Numeric;
use crate::op::{AddOp, BackwardArgs, ElemMulOp, ElemPowOp, ForwardInput, Op, ReLU, ScalarMulOp};
use crate::render::{Edge, Graphable, Node};
use crate::scalar::Scalar;
use crate::tensor::{BasicTensor, Tensor};
use num::One;
use std::cell::{Ref, RefCell};
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

pub type Id = u64;

thread_local!(static NEXT_ID: RefCell<Id> = const { RefCell::new(1) });

#[derive(Debug, Clone)]
pub enum Var<Tn: Tensor> {
    Parameter(Rc<RefCell<Param<Tn::T>>>, PhantomData<Tn>),
    Output(Rc<RefCell<Output<Tn::T>>>, PhantomData<Tn>),
}

#[derive(Debug, Clone)]
pub enum BackwardError {
    NonRootNode,
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
    children: Children,
    all_children: Option<HashMap<Id, VarRef<T>>>,
}

#[derive(Debug)]
enum Children {
    Unary(Id),
    Binary(Id, Id),
}

impl<Tn: Tensor> Var<Tn> {
    pub fn new(t: Tn) -> Self {
        let param = Param {
            id: Self::next_id(),
            data: Box::new(t),
            grad: None,
        };

        Self::Parameter(Rc::new(RefCell::new(param)), PhantomData)
    }

    pub fn id(&self) -> Id {
        match self {
            Self::Parameter(p, _) => p.borrow().id,
            Self::Output(o, _) => o.borrow().id,
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

    fn new_from_unary(&self, op: Box<dyn Op<Tn::T>>) -> Self {
        let self_ref = VarRef::from(self);
        let all_children = self_ref.take_all_children();
        let children = Children::Unary(self_ref.id());

        let self_data = self_ref.data();
        let data = op.forward(ForwardInput::Unary(self_data.as_ref()));
        let id = Self::next_id();

        let out = Output {
            id,
            data,
            op,
            children,
            all_children: Some(all_children),
        };

        let out_var = Self::Output(Rc::new(RefCell::new(out)), PhantomData);

        // Insert newly created ref into all_children
        let out_ref = VarRef::from(&out_var.clone());
        if let Self::Output(o, _) = &out_var {
            let mut out = o.borrow_mut();
            out.all_children.as_mut().unwrap().insert(id, out_ref);
        }

        out_var
    }

    fn new_from_binary(&self, other: VarRef<Tn::T>, op: Box<dyn Op<Tn::T>>) -> Self {
        let self_ref = VarRef::from(self);
        if self_ref.id() == other.id() {
            panic!("cannot use the same var for both arguments of a binary op");
        }

        let self_children = self_ref.take_all_children();
        let all_children = other.merge_all_children(self_children);
        let children = Children::Binary(self_ref.id(), other.id());

        let self_data = self_ref.data();
        let other_data = other.data();
        let data = op.forward(ForwardInput::Binary(
            self_data.as_ref(),
            other_data.as_ref(),
        ));

        let id = Self::next_id();
        let out = Output {
            id,
            data,
            op,
            children,
            all_children: Some(all_children),
        };
        let out_var = Self::Output(Rc::new(RefCell::new(out)), PhantomData);

        // Insert newly created ref into all_children
        let out_ref = VarRef::from(&out_var.clone());
        if let Self::Output(o, _) = &out_var {
            let mut out = o.borrow_mut();
            out.all_children.as_mut().unwrap().insert(id, out_ref);
        }

        out_var
    }
}

impl<T: Numeric> VarRef<T> {
    fn id(&self) -> Id {
        match self {
            Self::Parameter(p) => p.borrow().id,
            Self::Output(o) => o.borrow().id,
        }
    }

    fn backward(&mut self) -> Result<(), BackwardError> {
        match self.clone() {
            Self::Parameter(p) => {
                let mut param = p.borrow_mut();
                param.grad = Some(param.data.ones_with_shape());
                Ok(())
            }
            Self::Output(o) => {
                let out = o.borrow();
                let all_children = &out
                    .all_children
                    .as_ref()
                    .ok_or(BackwardError::NonRootNode)?;
                self.update_grads(all_children);
                Ok(())
            }
        }
    }

    fn take_all_children(&self) -> HashMap<Id, VarRef<T>> {
        match self {
            Self::Parameter(p) => {
                let mut blank_children = HashMap::new();
                blank_children.insert(p.borrow().id, self.clone());
                blank_children
            }
            Self::Output(o) => {
                let mut self_out = o.borrow_mut();
                self_out.all_children.take().unwrap_or_default()
            }
        }
    }

    fn merge_all_children(
        &self,
        mut all_children: HashMap<Id, VarRef<T>>,
    ) -> HashMap<Id, VarRef<T>> {
        match self {
            Self::Parameter(p) => {
                all_children.insert(p.borrow().id, self.clone());
                all_children
            }
            Self::Output(o) => {
                let mut self_out = o.borrow_mut();
                let mut self_children = self_out.all_children.take().unwrap_or_default();
                all_children.extend(self_children.drain());
                all_children
            }
        }
    }

    fn data(&self) -> Ref<Box<dyn BasicTensor<T>>> {
        match self {
            Self::Parameter(p) => Ref::map(p.borrow(), |p| &p.data),
            Self::Output(o) => Ref::map(o.borrow(), |o| &o.data),
        }
    }

    fn update_grads(&self, all_children: &HashMap<Id, VarRef<T>>) {
        let grads = self.calc_grads(all_children);
        for (id, grad) in grads.into_iter() {
            let v = &all_children[&id];
            match v {
                Self::Parameter(p) => {
                    let mut param = p.borrow_mut();
                    param.grad = Some(grad);
                }
                Self::Output(_) => {
                    // We just throw away non-param grads
                }
            }
        }
    }

    fn calc_grads(
        &self,
        all_children: &HashMap<Id, VarRef<T>>,
    ) -> HashMap<Id, Box<dyn BasicTensor<T>>> {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();
        Self::build_topo(self, &mut topo, &mut visited, all_children);

        let mut accumulators = HashMap::new();
        let ones = self.data().ones_with_shape();
        accumulators.insert(self.id(), ones);

        for v in topo.iter().rev() {
            match v {
                Self::Parameter(_) => {
                    // We're at a parameter which has no children; nothing left
                    // to do
                }
                Self::Output(o) => {
                    let out = o.borrow();
                    out.update_child_grads(&mut accumulators, all_children);
                }
            }
        }
        accumulators
    }

    fn children(&self, all_children: &HashMap<Id, VarRef<T>>) -> Vec<VarRef<T>> {
        match self {
            Self::Parameter(_) => vec![],
            Self::Output(o) => {
                let out = o.borrow();
                match &out.children {
                    Children::Unary(c) => vec![all_children[c].clone()],
                    Children::Binary(c1, c2) => {
                        vec![all_children[c1].clone(), all_children[c2].clone()]
                    }
                }
            }
        }
    }

    fn build_topo(
        cur: &VarRef<T>,
        topo: &mut Vec<VarRef<T>>,
        visited: &mut HashSet<Id>,
        all_children: &HashMap<Id, VarRef<T>>,
    ) {
        if visited.contains(&cur.id()) {
            return;
        }

        visited.insert(cur.id());
        for child in cur.children(all_children).iter() {
            Self::build_topo(child, topo, visited, all_children);
        }
        topo.push(cur.clone());
    }

    fn grad(
        &self,
        accumulators: &mut HashMap<Id, Box<dyn BasicTensor<T>>>,
    ) -> Box<dyn BasicTensor<T>> {
        accumulators
            .remove(&self.id())
            .unwrap_or_else(|| self.data().zeros_with_shape())
    }

    fn to_graph_node(&self, grads: &HashMap<Id, Box<dyn BasicTensor<T>>>) -> Node {
        let data = self.data();
        let grad = &grads[&self.id()];
        let label = format!(
            "{} | data: {} | grad: {}",
            self.id(),
            // TODO: non-scalars?
            data[&[]],
            grad[&[]],
        );
        let id = format!("{}", self.id());

        Node { id, label }
    }

    fn trace(&self) -> (HashSet<Node>, HashSet<Edge>) {
        match self {
            Self::Parameter(_) => panic!("can't trace a parameter"),
            Self::Output(o) => {
                let out = o.borrow();
                let mut nodes = HashSet::new();
                let mut edges = HashSet::new();
                let all_children = &out
                    .all_children
                    .as_ref()
                    .expect("Cannot call trace() on non-root node");

                Self::build_trace(self, &mut nodes, &mut edges, all_children);
                let grads = self.calc_grads(all_children);

                let g_nodes = nodes
                    .iter()
                    .map(|n| all_children[n].to_graph_node(&grads))
                    .collect();
                let g_edges = edges
                    .iter()
                    .map(|(from, to)| (format!("{}", from), format!("{}", to)))
                    .collect();

                (g_nodes, g_edges)
            }
        }
    }

    fn build_trace(
        val: &VarRef<T>,
        nodes: &mut HashSet<Id>,
        edges: &mut HashSet<(Id, Id)>,
        all_children: &HashMap<Id, VarRef<T>>,
    ) {
        if !nodes.contains(&val.id()) {
            nodes.insert(val.id());
            for child in val.children(all_children).iter() {
                edges.insert((child.id(), val.id()));
                Self::build_trace(child, nodes, edges, all_children);
            }
        }
    }
}

impl<T: Numeric> Output<T> {
    fn update_child_grads(
        &self,
        accumulators: &mut HashMap<Id, Box<dyn BasicTensor<T>>>,
        all_children: &HashMap<Id, VarRef<T>>,
    ) {
        match &self.children {
            Children::Unary(c_id) => {
                let c = all_children[c_id].clone();
                let in_grad = c.grad(accumulators);
                let in_data = c.data();
                let out_grad = accumulators
                    .get(&self.id)
                    .expect("expected out gradient to have been set");
                let args = BackwardArgs::Unary {
                    in_grad,
                    in_data: in_data.as_ref(),
                    out_grad: out_grad.as_ref(),
                    out_data: self.data.as_ref(),
                };
                let updated_grad = self.op.backward(args).unary();
                accumulators.insert(*c_id, updated_grad);
            }
            Children::Binary(c1_id, c2_id) => {
                let c1 = all_children[c1_id].clone();
                let c2 = all_children[c2_id].clone();
                let in_grad_1 = c1.grad(accumulators);
                let in_grad_2 = c2.grad(accumulators);
                let in_data_1 = c1.data();
                let in_data_2 = c2.data();
                let out_grad = accumulators
                    .get(&self.id)
                    .expect("expected out gradient to have been set");
                let args = BackwardArgs::Binary {
                    in_grad: (in_grad_1, in_grad_2),
                    in_data: (in_data_1.as_ref(), in_data_2.as_ref()),
                    out_grad: out_grad.as_ref(),
                    out_data: self.data.as_ref(),
                };
                let (updated_grad_1, updated_grad_2) = self.op.backward(args).binary();
                accumulators.insert(*c1_id, updated_grad_1);
                accumulators.insert(*c2_id, updated_grad_2);
            }
        }
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

impl<T: Numeric> From<T> for Var<Scalar<T>> {
    fn from(v: T) -> Self {
        Self::new(Scalar::from(v))
    }
}

impl<Tn: Tensor> Var<Tn> {
    pub fn data(&self) -> Ref<Tn> {
        match self {
            Self::Parameter(p, _) => Ref::map(p.borrow(), |p| {
                let data = Tn::ref_from_basic(p.data.as_ref());
                data
            }),
            Self::Output(o, _) => Ref::map(o.borrow(), |o| {
                let data = Tn::ref_from_basic(o.data.as_ref());
                data
            }),
        }
    }

    pub fn grad(&self) -> Option<Ref<Tn>> {
        match self {
            Self::Parameter(p, _) => {
                let param = p.borrow();
                // I can't for the life of me figure out how to get this to work
                // using Option#map
                #[allow(clippy::manual_map)]
                match param.grad {
                    Some(_) => Some(Ref::map(param, |p| {
                        let grad = Tn::ref_from_basic(p.grad.as_ref().unwrap().as_ref());
                        grad
                    })),
                    None => None,
                }
            }
            Self::Output(_, _) => None,
        }
    }

    pub fn relu(&self) -> Self {
        let op = ReLU::<Tn>::new();

        self.new_from_unary(op)
    }

    pub fn elem_pow(&self, n: Tn::T) -> Self {
        let op = ElemPowOp::<Tn>::new(n);

        self.new_from_unary(op)
    }

    pub fn elem_mul(&self, other: Var<Tn>) -> Self {
        if self.id() == other.id() {
            return self.elem_pow(Tn::T::two());
        }
        let op = ElemMulOp::<Tn>::new();
        let other_ref = (&other).into();

        self.new_from_binary(other_ref, op)
    }

    pub fn sum_elems(&self) -> Var<Scalar<Tn::T>> {
        self.data().iter().map(|(_, val)| Var::from(val)).sum()
    }
}

impl<Tn: Tensor> Sum for Var<Tn> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.reduce(|a, b| a + b)
            .unwrap_or_else(|| Var::new(Tn::zeros()))
    }
}

impl<T: Numeric> Var<Scalar<T>> {
    pub fn backward(&self) -> Result<(), BackwardError> {
        VarRef::from(self).backward()
    }
}

impl<Tn: Tensor> Add<Var<Tn>> for Var<Tn> {
    type Output = Self;

    fn add(self, other: Var<Tn>) -> Self {
        if self.id() == other.id() {
            return self * Var::from(Tn::T::two());
        }
        let op = AddOp::<Tn>::new();
        let other_ref: VarRef<Tn::T> = (&other).into();

        self.new_from_binary(other_ref, op)
    }
}

impl<Tn: Tensor> Mul<Var<Scalar<Tn::T>>> for Var<Tn> {
    type Output = Self;

    fn mul(self, other: Var<Scalar<Tn::T>>) -> Self {
        if self.id() == other.id() {
            return self.elem_pow(Tn::T::two());
        }
        let op = ScalarMulOp::<Tn>::new();
        let other_ref: VarRef<Tn::T> = (&other).into();

        self.new_from_binary(other_ref, op)
    }
}

impl<Tn: Tensor> Div<Var<Scalar<Tn::T>>> for Var<Tn> {
    type Output = Self;

    fn div(self, other: Var<Scalar<Tn::T>>) -> Self {
        let inv = other.elem_pow(-Tn::T::one());

        self * inv
    }
}

impl<Tn: Tensor> Neg for Var<Tn> {
    type Output = Self;

    fn neg(self) -> Self {
        self * Var::from(-Tn::T::one())
    }
}

impl<Tn: Tensor> Sub for Var<Tn> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

impl<T: Numeric> Graphable for Var<Scalar<T>> {
    fn trace(&self) -> (HashSet<Node>, HashSet<Edge>) {
        let val: VarRef<T> = self.into();
        val.trace()
    }
}

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;

    use super::*;

    #[test]
    fn test_equal() {
        let m: Matrix<f64, 3, 2> = Matrix::from([[1, 2], [3, 4], [5, 6]]);
        let v = Var::new(m);
        let s = v.sum_elems();

        assert_eq!(Scalar::from(1 + 2 + 3 + 4 + 5 + 6), *s.data());
    }
}
