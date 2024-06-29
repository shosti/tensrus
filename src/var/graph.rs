use std::collections::{HashMap, HashSet};

use super::{Children, Id, VarRef};

pub struct Graph {
    pub nodes: HashMap<Id, VarRef>,
    pub edges: HashMap<Id, Children>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
        }
    }

    /// Merges rhs into self. rhs will be empty after the operation is finished.
    pub fn merge(&mut self, rhs: &mut Self) {
        self.nodes.extend(rhs.nodes.drain());
        self.edges.extend(rhs.edges.drain());
    }

    pub fn insert(&mut self, var: VarRef, children: Option<Children>) {
        let id = var.id();
        self.nodes.insert(id, var);
        if let Some(children) = children {
            self.edges.insert(id, children);
        }
    }

    /// Build a topological sort of the graph starting at node with id ID
    pub fn topo(&self, root: Id) -> Vec<VarRef> {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();
        self.build_topo(root, &mut topo, &mut visited);

        topo.iter().map(|id| self.nodes[id].clone()).collect()
    }

    fn build_topo(&self, cur: Id, topo: &mut Vec<Id>, visited: &mut HashSet<Id>) {
        if visited.contains(&cur) {
            return;
        }

        visited.insert(cur);
        for &child in self.children(cur).iter() {
            self.build_topo(child, topo, visited);
        }
        topo.push(cur);
    }

    fn children(&self, id: Id) -> Vec<Id> {
        match self.edges.get(&id) {
            Some(Children::Unary(id)) => vec![*id],
            Some(Children::Binary(lhs, rhs)) => vec![*lhs, *rhs],
            None => vec![],
        }
    }
}

impl std::fmt::Debug for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Graph ({} nodes, {} edges)",
            self.nodes.len(),
            self.edges.len()
        )
    }
}
