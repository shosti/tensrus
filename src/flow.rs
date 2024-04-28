use crate::{
    numeric::Numeric,
    op::{NoOp, Op, PowOp},
    scalar::Scalar,
    tensor::TensorOps,
};
use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::HashSet,
    fmt::Debug,
    hash::{Hash, Hasher},
    rc::Rc,
};

thread_local!(static NEXT_ID: RefCell<u64> = const { RefCell::new(1) });

#[derive(Debug, Clone)]
pub struct Flow<T: Numeric, Tn: TensorOps<T>> {
    inner: Rc<RefCell<FlowInner<T, Tn>>>,
}

#[derive(Debug)]
struct FlowInner<T: Numeric, Tn: TensorOps<T>> {
    id: u64,
    data: Tn,
    grad: Tn,
    op: Box<dyn Op<T, Tn>>,
}

impl<T: Numeric, Tn: TensorOps<T>> Flow<T, Tn> {
    pub fn new(val: Tn) -> Self {
        let inner = Rc::new(RefCell::new(FlowInner {
            id: Self::next_id(),
            data: val,
            grad: Tn::zeros(),
            op: Box::new(NoOp {}),
        }));
        Self { inner }
    }

    pub fn new_from_op(val: Tn, op: impl Op<T, Tn> + 'static) -> Self {
        let inner = Rc::new(RefCell::new(FlowInner {
            id: Self::next_id(),
            data: val,
            grad: Tn::zeros(),
            op: Box::new(op),
        }));
        Self { inner }
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
        self.inner.borrow().id
    }

    pub fn op(&self) -> String {
        format!("{:?}", self.inner.borrow().op)
    }

    // pub fn update_from_grad(&self, epsilon: T) {
    //     let mut inner = self.inner.borrow_mut();
    //     let grad = inner.grad;

    //     inner.data += -epsilon * grad;
    // }

    pub fn update_grad(&self, f: impl Fn(T, T) -> T) {
        let inner_immut = self.inner.borrow();
        let mut inner = self.inner.borrow_mut();
        inner.grad.update_zip(&inner_immut.data, &f);
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
            for child in val.inner.borrow().op.children().iter() {
                edges.insert((child.clone(), val.clone()));
                Self::build_trace(child, nodes, edges);
            }
        }
    }
}

// impl<T: Numeric, const R: usize, const S: TensorShape, Tn: Tensor<T, R, S>> {
//     }

//     fn new_from_op(data: T, prev: HashSet<Flow<T>>, op: String) -> Self {
//         let mut children: Vec<Flow<T>> = prev.into_iter().collect();
//         children.sort_by(|v, t| {
//             v.val()
//                 .partial_cmp(&t.val())
//                 .unwrap_or(std::cmp::Ordering::Equal)
//         });
//         let inner = Rc::new(RefCell::new(FlowInner {
//             id: Self::next_id(),
//             data,
//             grad: T::zero(),
//             backward: None,
//             prev: children,
//             op,
//         }));

//         Self { inner }
//     }
//     // returns (nodes, edges)
//     pub fn pow(&self, n: T) -> Self {
//         let val = self.inner.borrow().data.powf(n);
//         let children = HashSet::from([self.clone()]);
//         let out = Self::new_from_op(val, children, "^".to_string());

//         let self_grad = self.clone();
//         let backward = move |grad, _| {
//             let data = self_grad.inner.borrow().data;
//             let mut self_inner = self_grad.inner.borrow_mut();
//             self_inner.grad += (n * data.powf(n - T::one())) * grad;
//         };
//         out.inner.borrow_mut().backward = Some(Box::new(backward));

//         out
//     }

//     pub fn relu(&self) -> Self {
//         let data = self.inner.borrow().data;
//         let outval = if data.is_sign_negative() {
//             T::zero()
//         } else {
//             data
//         };
//         let children = HashSet::from([self.clone()]);
//         let out = Self::new_from_op(outval, children, "ReLU".to_string());

//         let self_grad = self.clone();
//         let backward = move |grad, data: T| {
//             let mut self_inner = self_grad.inner.borrow_mut();
//             let diff = if data.is_sign_positive() && !data.is_zero() {
//                 grad
//             } else {
//                 T::zero()
//             };

//             self_inner.grad += diff;
//         };
//         out.inner.borrow_mut().backward = Some(Box::new(backward));

//         out
//     }

//     pub fn zero_grad(&self) {
//         self.inner.borrow_mut().grad = T::zero();
//     }
// }

impl<T: Numeric> Flow<T, Scalar<T>> {
    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        Self::build_topo(self, &mut topo, &mut visited);

        self.inner.borrow_mut().grad = Scalar::from(T::one());
        for flow in topo.iter().rev() {
            {
                let inner = flow.inner.borrow();
                inner.op.backward(&inner.grad, &inner.data);
            }
        }
    }

    fn build_topo(cur: &Self, topo: &mut Vec<Self>, visited: &mut HashSet<u64>) {
        let flow = cur.inner.borrow();
        if visited.contains(&flow.id) {
            return;
        }

        visited.insert(flow.id);
        for child in flow.op.children().iter() {
            Self::build_topo(child, topo, visited);
        }
        topo.push(cur.clone());
    }

    pub fn pow(&self, n: T) -> Self {
        PowOp::create_flow(self.clone(), n)
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
