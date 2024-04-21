use crate::numeric::Numeric;
use crate::value::Value;
use dot::{render, Edges, GraphWalk, Id, LabelText, Labeller, Nodes, RankDir};
use std::borrow::Cow;
use std::collections::hash_map::HashMap;
use std::io::Write;

#[derive(Clone)]
struct Nd {
    id: String,
    label: String,
}
type Ed = (String, String);

pub struct Graph {
    nodes: HashMap<String, Nd>,
    edges: Vec<Ed>,
}

impl Graph {
    pub fn new<T: Numeric>(val: Value<T>) -> Self {
        let (nodes, edges) = val.trace();

        let mut ns = HashMap::new();
        let mut es = Vec::new();

        for n in nodes.iter() {
            let label = match n.op() {
                Some(op) => format!("{} | {} | data {:.4} | grad {:.4}", n.id(), op, n.val(), n.grad()),
                None => format!("{} | data {:.4} | grad {:.4}", n.id(), n.val(), n.grad()),
            };
            let id = format!("id{}", n.id());
            let node = Nd {
                id: id.clone(),
                label,
            };
            ns.insert(id, node);
        }

        for (from, to) in edges.iter() {
            let edge = (format!("id{}", from.id()), format!("id{}", to.id()));
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

impl<'a> Labeller<'a, Nd, Ed> for Graph {
    fn graph_id(&'a self) -> Id<'a> {
        Id::new("valuegraph").unwrap()
    }

    fn node_id(&self, n: &Nd) -> Id {
        Id::new(n.id.clone()).unwrap()
    }

    fn node_label(&'a self, n: &Nd) -> LabelText<'a> {
        LabelText::LabelStr(Cow::Owned(n.label.clone()))
    }

    fn node_shape(&'a self, _: &Nd) -> Option<LabelText<'a>> {
        Some(LabelText::LabelStr(Cow::Owned("record".to_string())))
    }
}

impl<'a> GraphWalk<'a, Nd, Ed> for Graph {
    fn nodes(&'a self) -> Nodes<'a, Nd> {
        let nodes: Vec<Nd> = self.nodes.values().map(|n| n.clone()).collect();
        Cow::Owned(nodes)
    }

    fn edges(&'a self) -> Edges<'a, Ed> {
        Cow::Borrowed(&self.edges[..])
    }

    fn source(&'a self, e: &Ed) -> Nd {
        self.nodes.get(&e.0).unwrap().clone()
    }
    fn target(&'a self, e: &Ed) -> Nd {
        self.nodes.get(&e.1).unwrap().clone()
    }
}
