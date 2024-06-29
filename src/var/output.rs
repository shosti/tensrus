use super::graph::Graph;
use super::var_ref::VarRef;
use super::{Children, Id};
use crate::differentiable::{Differentiable, DifferentiableTensor};
use crate::errors::GraphError;
use crate::op::{BackwardArgs, ForwardInput};
use crate::{dyn_tensor::DynTensor, op::Op};
use std::assert_matches::assert_matches;
use std::cell::Ref;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::rc::Weak;
use std::{cell::RefCell, rc::Rc};

#[derive(Debug, Clone)]
pub struct Output<Tn> {
    pub(super) inner: Rc<RefCell<OutputInner>>,
    marker: PhantomData<Tn>,
}

#[derive(Debug)]
pub(super) struct OutputInner {
    pub(super) id: Id,
    pub(super) graph: GraphRef,
    pub(super) data: Option<Box<dyn DynTensor>>,
    op: Box<dyn Op>,
    children: Children,
}

#[derive(Debug, Clone)]
pub(super) enum GraphRef {
    Root(Rc<RefCell<Graph>>),
    NonRoot(Weak<RefCell<Graph>>),
}

impl<Tn> Output<Tn>
where
    Tn: DifferentiableTensor,
    Tn::T: Differentiable,
{
    pub fn id(&self) -> Id {
        self.inner.borrow().id
    }

    pub fn data(&self) -> Ref<Tn> {
        self.forward().unwrap();

        Ref::map(self.inner.borrow(), |inner| {
            if let Some(ref data) = inner.data {
                Tn::ref_from_dyn(data.as_ref())
            } else {
                panic!("output data not calculated")
            }
        })
    }

    pub fn into_data(self) -> Option<Tn> {
        self.forward().ok()?;

        let mut inner = self.inner.borrow_mut();
        let data = inner.data.take()?;
        Some(*Tn::from_dyn(data))
    }

    pub(super) fn new_unary(id: Id, op: Box<dyn Op>, child: VarRef) -> Self {
        let children = Children::Unary(child.id());
        let graph = Self::build_graph_unary(&child);
        let inner = OutputInner::new(id, op, children, graph);

        Self {
            inner,
            marker: PhantomData,
        }
    }

    pub(super) fn new_binary(id: Id, op: Box<dyn Op>, lhs: VarRef, rhs: VarRef) -> Self {
        let graph = Self::build_graph_binary(&lhs, &rhs);
        let children = Children::Binary(lhs.id(), rhs.id());
        let inner = OutputInner::new(id, op, children, graph);

        Self {
            inner,
            marker: PhantomData,
        }
    }

    fn build_graph_unary(child: &VarRef) -> GraphRef {
        let root = child.graph_root().unwrap();

        root.take_graph()
    }

    fn build_graph_binary(lhs: &VarRef, rhs: &VarRef) -> GraphRef {
        let lhs_root = lhs.graph_root().unwrap();
        let rhs_root = rhs.graph_root().unwrap();
        if lhs_root.id() == rhs_root.id() {
            // We're part of the same graph (which implies that lhs and rhs are
            // both Outputs), so we treat lhs_root as a unary graph.
            assert_matches!(lhs_root, VarRef::Output(_));
            return lhs_root.take_graph();
        }

        let lhs_graph = lhs_root.take_graph();
        let rhs_graph = rhs_root.take_graph();

        // Point all nodes in rhs_graph to lhs_graph
        for node in rhs_graph.nodes().unwrap() {
            if let VarRef::Output(inner_ref) = node {
                let mut inner = inner_ref.borrow_mut();
                assert_matches!(inner.graph, GraphRef::NonRoot(_));
                inner.graph = lhs_graph.downgrade();
            }
        }

        // Merge lhs and rhs graphs
        match (&lhs_graph, &rhs_graph) {
            (GraphRef::Root(ref lhs_ref), GraphRef::Root(ref rhs_ref)) => {
                let mut lhs_graph = lhs_ref.borrow_mut();
                let mut rhs_graph = rhs_ref.borrow_mut();
                lhs_graph.merge(&mut rhs_graph);
            }
            _ => panic!("expected root graphs"),
        }

        lhs_graph
    }

    pub(super) fn backward(&self) -> Result<(), GraphError> {
        // First we do the forward pass to make sure data is populated.
        self.forward()?;

        let graph = self.inner.borrow().graph.clone();
        let grads = self.inner.borrow().calc_grads()?;
        for (id, grad) in grads.into_iter() {
            match graph.get(&id)? {
                VarRef::Param(p) => {
                    let mut param = p.borrow_mut();
                    param.set_grad(grad);
                }
                VarRef::Input(_) | VarRef::Output(_) => {} // Discard non-param grads
            }
        }
        Ok(())
    }

    fn forward(&self) -> Result<(), GraphError> {
        let id = self.id();
        let graph = self.inner.borrow().graph.clone();

        let children = graph.topo(id)?;

        for child in children.iter() {
            if let VarRef::Output(inner_ref) = child {
                let mut inner = inner_ref.borrow_mut();
                inner.forward()?;
            }
        }

        Ok(())
    }
}

impl OutputInner {
    fn new(id: Id, op: Box<dyn Op>, children: Children, graph: GraphRef) -> Rc<RefCell<Self>> {
        let inner = Rc::new(RefCell::new(Self {
            id,
            graph: graph.clone(),
            data: None,
            op,
            children,
        }));
        let var = VarRef::Output(inner.clone());
        graph.insert(var, Some(children)).unwrap();

        inner
    }

    fn forward(&mut self) -> Result<(), GraphError> {
        if self.data.is_some() {
            return Ok(());
        }

        match self.children {
            Children::Unary(id) => {
                let child = self.graph.get(&id)?;
                let child_data = child.data();

                let args = ForwardInput::Unary(child_data.as_ref());
                let data = self.op.forward(args);

                self.data = Some(data);
            }
            Children::Binary(lhs_id, rhs_id) => {
                let lhs = self.graph.get(&lhs_id)?;
                let lhs_data = lhs.data();
                let rhs = self.graph.get(&rhs_id)?;
                let rhs_data = rhs.data();

                let args = ForwardInput::Binary(lhs_data.as_ref(), rhs_data.as_ref());
                let data = self.op.forward(args);

                self.data = Some(data);
            }
        }

        Ok(())
    }

    fn calc_grads(&self) -> Result<HashMap<Id, Box<dyn DynTensor>>, GraphError> {
        let topo = self.graph.topo(self.id)?;
        let mut accumulators = HashMap::new();

        // Root node (self) has dy/dl of 1; insert that into the accumulators to
        // bootstrap the backward pass
        let ones = self
            .data
            .as_ref()
            .expect("output data should have been populated during the forward pass")
            .ones_with_shape();
        accumulators.insert(self.id, ones);

        for v in topo.iter().rev() {
            match v {
                VarRef::Param(_) | VarRef::Input(_) => {
                    // We're at the edges of the graph; nothing left to do
                }
                VarRef::Output(inner) => {
                    let o = inner.borrow();
                    o.update_child_grads(&mut accumulators)?;
                }
            }
        }

        Ok(accumulators)
    }

    fn update_child_grads(
        &self,
        accumulators: &mut HashMap<Id, Box<dyn DynTensor>>,
    ) -> Result<(), GraphError> {
        match &self.children {
            Children::Unary(c_id) => {
                let c = self.graph.get(c_id)?.clone();
                let in_grad = c.grad(accumulators);
                let in_data = c.data();
                let out_grad = accumulators
                    .get(&self.id)
                    .expect("expected out gradient to have been set");
                let args = BackwardArgs::Unary {
                    in_grad,
                    in_data: in_data.as_ref(),
                    out_grad: out_grad.as_ref(),
                    out_data: self
                        .data
                        .as_ref()
                        .expect("expected data to have been calculated during the forward pass")
                        .as_ref(),
                };
                let updated_grad = self.op.backward(args).unary();
                accumulators.insert(*c_id, updated_grad);
            }
            Children::Binary(c1_id, c2_id) => {
                let c1 = self.graph.get(c1_id)?.clone();
                let c2 = self.graph.get(c2_id)?.clone();
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
                    out_data: self
                        .data
                        .as_ref()
                        .expect("expected data to have been calculated during the forward pass")
                        .as_ref(),
                };
                let (updated_grad_1, updated_grad_2) = self.op.backward(args).binary();
                accumulators.insert(*c1_id, updated_grad_1);
                accumulators.insert(*c2_id, updated_grad_2);
            }
        }

        Ok(())
    }

    /// Returns the ID of the Output that contains the root reference to the
    /// graph.
    pub(super) fn graph_root_id(&self) -> Id {
        let graph_ref = match &self.graph {
            GraphRef::Root(_) => return self.id,
            GraphRef::NonRoot(graph) => graph.clone(),
        };

        for (&id, var) in graph_ref.upgrade().unwrap().borrow().nodes.iter() {
            if let VarRef::Output(inner_ref) = var {
                let inner = inner_ref.borrow();
                if inner.graph.is_root() {
                    return id;
                }
            }
        }

        panic!("no root ID found in graph")
    }
}

impl GraphRef {
    fn insert(&self, var: VarRef, children: Option<Children>) -> Result<(), GraphError> {
        match self {
            GraphRef::Root(graph) => {
                graph.borrow_mut().insert(var, children);
            }
            GraphRef::NonRoot(graph) => {
                graph
                    .upgrade()
                    .ok_or(GraphError::GraphDropped)?
                    .borrow_mut()
                    .insert(var, children);
            }
        }

        Ok(())
    }

    fn nodes(&self) -> Result<Vec<VarRef>, GraphError> {
        let n = match self {
            GraphRef::Root(graph) => graph.borrow().nodes.values().cloned().collect(),
            GraphRef::NonRoot(graph) => graph
                .upgrade()
                .ok_or(GraphError::GraphDropped)?
                .borrow()
                .nodes
                .values()
                .cloned()
                .collect(),
        };
        Ok(n)
    }

    fn topo(&self, root: Id) -> Result<Vec<VarRef>, GraphError> {
        let t = match self {
            GraphRef::Root(graph) => graph.borrow().topo(root),
            GraphRef::NonRoot(graph) => graph
                .upgrade()
                .ok_or(GraphError::GraphDropped)?
                .borrow()
                .topo(root),
        };
        Ok(t)
    }

    pub(super) fn get(&self, id: &Id) -> Result<VarRef, GraphError> {
        let val = match self {
            GraphRef::Root(graph) => graph.borrow().nodes[id].clone(),
            GraphRef::NonRoot(graph) => graph
                .upgrade()
                .ok_or(GraphError::GraphDropped)?
                .borrow()
                .nodes[id]
                .clone(),
        };

        Ok(val)
    }

    /// Downgrades a root GraphRef to a non-root one.
    pub(super) fn downgrade(&self) -> GraphRef {
        if let GraphRef::Root(graph) = self {
            GraphRef::NonRoot(Rc::downgrade(graph))
        } else {
            panic!("called downgrade on a non-root graph reference");
        }
    }

    fn is_root(&self) -> bool {
        matches!(self, GraphRef::Root(_))
    }
}

impl<Tn> std::ops::Drop for Output<Tn> {
    fn drop(&mut self) {
        // We have an invariant that every Graph has a single strong RC
        // reference to it (through a GraphRef::Root). Graphs hold strong RC
        // references to each node in the graph. This means that a cycle exists
        // as long as the Graph hasn't been dropped:
        //
        // - The Graph holds a strong RC to the OutputInner with the root GraphRef
        // - The OutputInner holds a strong RC to the Graph
        //
        // To break the cycle, if we're the last Output to hold a reference to
        // the OutputInner that has the last Graph reference, we downgrade the
        // reference so the whole Graph gets dropped.
        //
        // We know we're the last Output to hold a reference to the OutputInner
        // if we have a reference count of 2 (one reference from the Output and
        // one from the Graph).
        let inner_ref_count = Rc::strong_count(&self.inner);

        if inner_ref_count == 2 && self.inner.borrow().graph.is_root() {
            let mut inner = self.inner.borrow_mut();
            let g = inner.graph.clone();
            inner.graph = g.downgrade();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{scalar::Scalar, var::Var};

    #[test]
    fn test_mem_management() {
        let x = Var::param(Scalar::from(2.0));
        let y = Var::param(Scalar::from(3.0));
        let z = x + y;

        {
            let l = z.elem_pow(2.0);
            let _l2 = l.clone();
        }

        assert_eq!(z.backward().err().unwrap(), GraphError::GraphDropped);
    }
}
