use crate::{dyn_tensor::DynTensor, errors::GraphError};

use super::{
    graph::Graph,
    input::InputInner,
    output::{GraphRef, OutputInner},
    param::ParamInner,
    Id,
};
use std::{
    cell::{Ref, RefCell},
    collections::HashMap,
    rc::Rc,
};

#[derive(Debug, Clone)]
pub enum VarRef {
    Param(Rc<RefCell<ParamInner>>),
    Input(Rc<RefCell<InputInner>>),
    Output(Rc<RefCell<OutputInner>>),
}

impl VarRef {
    pub fn id(&self) -> Id {
        match self {
            Self::Param(inner) => inner.borrow().id,
            Self::Input(inner) => inner.borrow().id,
            Self::Output(inner) => inner.borrow().id,
        }
    }

    pub fn data(&self) -> Ref<Box<dyn DynTensor>> {
        match self {
            Self::Param(inner) => Ref::map(inner.borrow(), |inner| &inner.data),
            Self::Input(inner) => Ref::map(inner.borrow(), |inner| &inner.data),
            Self::Output(inner) => Ref::map(inner.borrow(), |inner| inner.data.as_ref().unwrap()),
        }
    }

    pub(super) fn grad(
        &self,
        accumulators: &mut HashMap<Id, Box<dyn DynTensor>>,
    ) -> Box<dyn DynTensor> {
        accumulators
            .remove(&self.id())
            .unwrap_or_else(|| self.data().zeros_with_shape())
    }

    pub(super) fn graph_root(&self) -> Result<VarRef, GraphError> {
        let root = match self {
            Self::Output(inner_ref) => {
                let inner = inner_ref.borrow();
                let root_id = inner.graph_root_id();
                inner.graph.get(&root_id)?.clone()
            }
            _ => self.clone(),
        };

        Ok(root)
    }

    pub(super) fn take_graph(&self) -> GraphRef {
        match self {
            Self::Output(inner_ref) => {
                let mut inner = inner_ref.borrow_mut();
                let g = inner.graph.clone();
                inner.graph = g.downgrade();
                g
            }
            _ => {
                let mut g = Graph::new();
                g.insert(self.clone(), None);
                GraphRef::Root(Rc::new(RefCell::new(g)))
            }
        }
    }
}
