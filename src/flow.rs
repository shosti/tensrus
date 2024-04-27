use crate::{
    numeric::Numeric,
    op::Op,
    tensor::{Tensor, TensorOps, TensorShape},
};
use std::{cell::RefCell, cmp::Ordering, fmt::Debug, rc::Rc};

thread_local!(static NEXT_ID: RefCell<u64> = const { RefCell::new(1) });

#[derive(Debug)]
pub struct Flow<T: Numeric, const R: usize, const S: TensorShape, Tn>
where
    Tn: Tensor<T, R, S> + TensorOps<T>,
{
    inner: Rc<RefCell<FlowInner<T, R, S, Tn>>>,
}

#[derive(Debug)]
struct FlowInner<T: Numeric, const R: usize, const S: TensorShape, Tn>
where
    Tn: Tensor<T, R, S> + TensorOps<T>,
{
    id: u64,
    data: Tn,
    grad: T,
    _op: Op<T, R, S, Tn>,
}
impl<T: Numeric, const R: usize, const S: TensorShape, Tn> Flow<T, R, S, Tn>
where
    Tn: Tensor<T, R, S> + TensorOps<T>,
{
    pub fn new(val: Tn) -> Self {
        let inner = Rc::new(RefCell::new(FlowInner {
            id: Self::next_id(),
            data: val,
            grad: T::zero(),
            _op: Op::None,
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

    pub fn grad(&self) -> T {
        self.inner.borrow().grad
    }

    pub fn update_from_grad(&self, epsilon: T) {
        let mut inner = self.inner.borrow_mut();
        let grad = inner.grad;

        inner.data += -epsilon * grad;
    }

    // fn build_topo(cur: &Flow<T>, topo: &mut Vec<Self>, visited: &mut HashSet<u64>) {
    //     let val = cur.inner.borrow();
    //     if visited.contains(&val.id) {
    //         return;
    //     }

    //     visited.insert(val.id);
    //     for child in val.prev.iter() {
    //         Self::build_topo(child, topo, visited);
    //     }
    //     topo.push(cur.clone());
    // }
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
//     pub fn trace(&self) -> (HashSet<Self>, HashSet<(Self, Self)>) {
//         let mut nodes = HashSet::new();
//         let mut edges = HashSet::new();

//         Self::build_trace(self, &mut nodes, &mut edges);

//         return (nodes, edges);
//     }

//     fn build_trace(val: &Self, nodes: &mut HashSet<Self>, edges: &mut HashSet<(Self, Self)>) {
//         if !nodes.contains(val) {
//             nodes.insert(val.clone());
//             for child in val.inner.borrow().prev.iter() {
//                 edges.insert((child.clone(), val.clone()));
//                 Self::build_trace(child, nodes, edges);
//             }
//         }
//     }

//     pub fn backward(&self) {
//         let mut topo = Vec::new();
//         let mut visited = HashSet::new();

//         Self::build_topo(self, &mut topo, &mut visited);

//         self.inner.borrow_mut().grad = T::one();
//         for val in topo.iter().rev() {
//             let grad;
//             let data;
//             {
//                 let inner = val.inner.borrow();
//                 grad = inner.grad;
//                 data = inner.data;
//             }
//             if let Some(backward) = &mut val.inner.borrow_mut().backward {
//                 backward(grad, data);
//             }
//         }
//     }

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

impl<T: Numeric, const R: usize, const S: TensorShape, Tn> Clone for Flow<T, R, S, Tn>
where
    Tn: Tensor<T, R, S> + TensorOps<T>,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T: Numeric, const R: usize, const S: TensorShape, Tn> PartialEq for Flow<T, R, S, Tn>
where
    Tn: Tensor<T, R, S> + TensorOps<T>,
{
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl<T: Numeric, const R: usize, const S: TensorShape, Tn> Eq for Flow<T, R, S, Tn> where
    Tn: Tensor<T, R, S> + TensorOps<T>
{
}

impl<T: Numeric, const R: usize, const S: TensorShape, Tn> PartialOrd for Flow<T, R, S, Tn>
where
    Tn: Tensor<T, R, S> + TensorOps<T>,
{
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

impl<T: Numeric, const R: usize, const S: TensorShape, Tn> Ord for Flow<T, R, S, Tn>
where
    Tn: Tensor<T, R, S> + TensorOps<T>,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}
