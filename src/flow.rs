use crate::{
    numeric::Numeric,
    op::{AddOp, MulOp, NoOp, Op, PowOp, ReluOp},
    scalar::Scalar,
    tensor::{BasicTensor, Tensor},
};
use std::{
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
pub struct Flow<T: Numeric, Tn: Tensor<T>> {
    pub id: u64,
    pub data: Tn,
    pub grad: Tn,
    op: Rc<RefCell<dyn Op<T>>>,
}

#[derive(Debug, Clone)]
pub struct FlowRef<T: Numeric> {
    pub id: u64,
    pub data: Rc<dyn BasicTensor<T>>,
    pub grad: Rc<dyn BasicTensor<T>>,
    op: Rc<RefCell<dyn Op<T>>>,
}

impl<T: Numeric, Tn: Tensor<T>> Flow<T, Tn> {
    pub fn new(data: Tn) -> Self {
        Self {
            id: Self::next_id(),
            data,
            grad: Tn::zeros(),
            op: Rc::new(RefCell::new(NoOp {})),
        }
    }

    pub fn new_from_op(data: Tn, op: impl Op<T>) -> Self {
        Self {
            id: Self::next_id(),
            data,
            grad: Tn::zeros(),
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

    pub fn update_from_grad(&self, epsilon: T) {
        self.data
            .clone()
            .update_zip(&self.grad, |data, grad| data + grad * -epsilon);
    }

    pub fn zero_grad(&mut self) {
        self.grad.update(|_| T::zero());
    }

    pub fn update_grad(&mut self, f: impl Fn(T, T) -> T) {
        self.grad.update_zip(&self.data, f);
    }

    // returns (nodes, edges)
    pub fn trace(&self) -> (HashSet<FlowRef<T>>, HashSet<(FlowRef<T>, FlowRef<T>)>) {
        let mut nodes = HashSet::new();
        let mut edges = HashSet::new();
        let val: FlowRef<T> = self.clone().into();

        Self::build_trace(&val, &mut nodes, &mut edges);

        (nodes, edges)
    }

    fn build_trace(val: &FlowRef<T>, nodes: &mut HashSet<FlowRef<T>>, edges: &mut HashSet<(FlowRef<T>, FlowRef<T>)>) {
        if !nodes.contains(val) {
            nodes.insert(val.clone());
            for child in val.op.borrow().children().iter() {
                edges.insert((child.clone(), val.clone()));
                Self::build_trace(child, nodes, edges);
            }
        }
    }
}

impl<T: Numeric> From<T> for Flow<T, Scalar<T>> {
    fn from(val: T) -> Self {
        Flow::new(Scalar::from(val))
    }
}

impl<T: Numeric> Flow<T, Scalar<T>> {
    pub fn backward(&mut self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();
        let cur: FlowRef<T> = self.clone().into();

        Self::build_topo(&cur, &mut topo, &mut visited);

        self.grad.update(|_| T::one());
        for flow in topo.iter().rev() {
            flow.op
                .borrow_mut()
                .backward(&flow);
        }
    }

    fn build_topo(cur: &FlowRef<T>, topo: &mut Vec<FlowRef<T>>, visited: &mut HashSet<u64>) {
        if visited.contains(&cur.id) {
            return;
        }

        visited.insert(cur.id);
        for child in cur.op.borrow().children().iter() {
            Self::build_topo(child, topo, visited);
        }
        topo.push(cur.clone());
    }

    pub fn pow(&self, n: T) -> Self {
        PowOp::create_flow(self.clone(), n)
    }
}

impl<T: Numeric, Tn> Add for Flow<T, Tn>
where
    Tn: Tensor<T> + Add<Output = Tn>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        AddOp::create_flow(self, other)
    }
}

impl<T: Numeric> Mul for Flow<T, Scalar<T>> {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        MulOp::create_flow(self, other)
    }
}

impl<T: Numeric> Div for Flow<T, Scalar<T>> {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        let inv = other.pow(-T::one());

        self * inv
    }
}

impl<T: Numeric> Sub for Flow<T, Scalar<T>> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self + (-other)
    }
}

impl<T: Numeric> Neg for Flow<T, Scalar<T>> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * Flow::new(Scalar::from(-T::one()))
    }
}

impl<T: Numeric, Tn: Tensor<T>> PartialEq for Flow<T, Tn> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T: Numeric, Tn: Tensor<T>> Eq for Flow<T, Tn> {}

impl<T: Numeric> FlowRef<T> {
    pub fn data<Tn: Tensor<T>>(&self) -> &Tn {
        let data: &Tn = self.data.as_any().downcast_ref().unwrap();
        data
    }

    pub fn grad<Tn: Tensor<T>>(&self) -> &Tn {
        let grad: &Tn = self.grad.as_any().downcast_ref().unwrap();
        grad
    }

    pub fn op(&self) -> String {
        format!("{:?}", self.op.borrow())
    }
}

impl<T: Numeric> Hash for FlowRef<T> {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.id.hash(state)
    }
}

impl<T: Numeric> PartialOrd for FlowRef<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.id.cmp(&other.id))
    }
}

impl<T: Numeric> Ord for FlowRef<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<T: Numeric> PartialEq for FlowRef<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T: Numeric> Eq for FlowRef<T> {}

impl<T: Numeric> Sum for Flow<T, Scalar<T>> {
    fn sum<I: Iterator<Item = Flow<T, Scalar<T>>>>(mut iter: I) -> Self {
        let first = iter.next();
        if first.is_none() {
            return Flow::from(T::zero());
        }
        let mut res = first.unwrap();

        for next in iter {
            res = res + next;
        }

        res
    }
}

impl<T: Numeric, Tn: Tensor<T>> From<Flow<T, Tn>> for FlowRef<T> {
    fn from(flow: Flow<T, Tn>) -> Self {
        Self {
            id: flow.id,
            data: Rc::new(flow.data.clone()),
            grad: Rc::new(flow.grad.clone()),
            op: flow.op.clone(),
        }
    }
}
