use crate::{
    numeric::Numeric,
    op::{AddOp, MulOp, NoOp, Op, PowOp, ReluOp},
    scalar::Scalar,
    tensor::TensorOps,
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
pub struct Flow<T: Numeric, Tn: TensorOps<T>> {
    id: u64,
    data: Tn,
    grad: Tn,
    op: Rc<RefCell<dyn Op<T, Tn>>>,
}

impl<T: Numeric, Tn: TensorOps<T>> Flow<T, Tn> {
    pub fn new(data: Tn) -> Self {
        Self {
            id: Self::next_id(),
            data,
            grad: Tn::zeros(),
            op: Rc::new(RefCell::new(NoOp {})),
        }
    }

    pub fn new_from_op(data: Tn, op: impl Op<T, Tn> + 'static) -> Self {
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

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn op(&self) -> String {
        format!("{:?}", self.op.borrow())
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
    pub fn trace(&self) -> (HashSet<Self>, HashSet<(Self, Self)>) {
        let mut nodes = HashSet::new();
        let mut edges = HashSet::new();

        Self::build_trace(self, &mut nodes, &mut edges);

        (nodes, edges)
    }

    fn build_trace(val: &Self, nodes: &mut HashSet<Self>, edges: &mut HashSet<(Self, Self)>) {
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
    pub fn val(&self) -> T {
        self.data.val()
    }

    pub fn grad(&self) -> T {
        self.grad.val()
    }
}

impl<T: Numeric> Flow<T, Scalar<T>> {
    pub fn backward(&mut self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        Self::build_topo(self, &mut topo, &mut visited);

        self.grad.update(|_| T::one());
        for flow in topo.iter().rev() {
            flow.op.borrow_mut().backward(&flow.grad, &flow.data);
        }
    }

    fn build_topo(cur: &Self, topo: &mut Vec<Self>, visited: &mut HashSet<u64>) {
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

    pub fn relu(&self) -> Self {
        ReluOp::create_flow(self.clone())
    }
}

impl<T: Numeric> Add for Flow<T, Scalar<T>> {
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

impl<T: Numeric, Tn: TensorOps<T>> PartialEq for Flow<T, Tn> {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl<T: Numeric, Tn: TensorOps<T>> Eq for Flow<T, Tn> {}

impl<T: Numeric, Tn: TensorOps<T>> Hash for Flow<T, Tn> {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.id().hash(state)
    }
}

impl<T: Numeric, Tn: TensorOps<T>> PartialOrd for Flow<T, Tn> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.id() < other.id() {
            Some(Ordering::Less)
        } else if self.id() == other.id() {
            Some(Ordering::Equal)
        } else {
            Some(Ordering::Greater)
        }
    }
}

impl<T: Numeric, Tn: TensorOps<T>> Ord for Flow<T, Tn> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

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
