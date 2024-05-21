use dot::{render, Edges, GraphWalk, Id, LabelText, Labeller, Nodes, RankDir};
use std::borrow::Cow;
use std::collections::hash_map::HashMap;
use std::collections::HashSet;
use std::io::Write;

pub type NodeId = String;

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct Node {
    pub id: NodeId,
    pub label: String,
}
pub type Edge = (NodeId, NodeId);

pub struct Graph {
    nodes: HashMap<String, Node>,
    edges: Vec<Edge>,
}

pub trait Graphable {
    fn trace(&self) -> (HashSet<Node>, HashSet<Edge>);
}

impl Graph {
    pub fn new(val: &dyn Graphable) -> Self {
        let (nodes, edges) = val.trace();

        let mut ns = HashMap::new();
        let mut es = Vec::new();

        for n in nodes.iter() {
            let id = format!("id{}", n.id);
            let node = Node {
                id: id.clone(),
                label: n.label.clone(),
            };
            ns.insert(id, node);
        }

        for (from, to) in edges.iter() {
            let edge = (format!("id{}", from), format!("id{}", to));
            es.push(edge);
        }

        Self {
            nodes: ns,
            edges: es,
        }
    }

    pub fn render_to<W: Write>(&self, output: &mut W) {
        render(self, output).unwrap();
    }
}

impl<'a> Labeller<'a, Node, Edge> for Graph {
    fn graph_id(&'a self) -> Id<'a> {
        Id::new("valuegraph").unwrap()
    }

    fn node_id(&self, n: &Node) -> Id {
        Id::new(n.id.clone()).unwrap()
    }

    fn node_label(&'a self, n: &Node) -> LabelText<'a> {
        LabelText::LabelStr(Cow::Owned(n.label.clone()))
    }

    fn node_shape(&'a self, _: &Node) -> Option<LabelText<'a>> {
        Some(LabelText::LabelStr(Cow::Owned("record".to_string())))
    }

    fn rank_dir(&'a self) -> Option<RankDir> {
        Some(RankDir::LeftRight)
    }
}

impl<'a> GraphWalk<'a, Node, Edge> for Graph {
    fn nodes(&'a self) -> Nodes<'a, Node> {
        let nodes: Vec<Node> = self.nodes.values().cloned().collect();
        Cow::Owned(nodes)
    }

    fn edges(&'a self) -> Edges<'a, Edge> {
        Cow::Borrowed(&self.edges[..])
    }

    fn source(&'a self, e: &Edge) -> Node {
        self.nodes.get(&e.0).unwrap().clone()
    }
    fn target(&'a self, e: &Edge) -> Node {
        self.nodes.get(&e.1).unwrap().clone()
    }
}
