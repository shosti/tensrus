// use crate::numeric::Numeric;
// use crate::scalar::Scalar;
// use crate::var::Var;
// use dot::{render, Edges, GraphWalk, Id, LabelText, Labeller, Nodes, RankDir};
// use std::borrow::Cow;
// use std::collections::hash_map::HashMap;
// use std::io::Write;

// #[derive(Clone)]
// struct Nd {
//     id: String,
//     label: String,
// }
// type Ed = (String, String);

// pub struct Graph {
//     nodes: HashMap<String, Nd>,
//     edges: Vec<Ed>,
// }

// impl Graph {
//     pub fn new<T: Numeric>(val: Var<Scalar<T>>) -> Self {
//         let (nodes, edges) = val.trace();

//         let mut ns = HashMap::new();
//         let mut es = Vec::new();

//         for n in nodes.iter() {
//             let data: &Scalar<T> = n.data();
//             let grad: &Scalar<T> = n.grad();
//             let label = format!(
//                 "{} | data: {:?} | grad: {:?} | {:?}",
//                 n.id,
//                 data,
//                 grad,
//                 n.op()
//             );
//             let id = format!("id{}", n.id);
//             let node = Nd {
//                 id: id.clone(),
//                 label,
//             };
//             ns.insert(id, node);
//         }

//         for (from, to) in edges.iter() {
//             let edge = (format!("id{}", from.id), format!("id{}", to.id));
//             es.push(edge);
//         }

//         Self {
//             nodes: ns,
//             edges: es,
//         }
//     }

//     pub fn render_to<W: Write>(&self, output: &mut W) {
//         render(self, output).unwrap();
//     }
// }

// impl<'a> Labeller<'a, Nd, Ed> for Graph {
//     fn graph_id(&'a self) -> Id<'a> {
//         Id::new("valuegraph").unwrap()
//     }

//     fn node_id(&self, n: &Nd) -> Id {
//         Id::new(n.id.clone()).unwrap()
//     }

//     fn node_label(&'a self, n: &Nd) -> LabelText<'a> {
//         LabelText::LabelStr(Cow::Owned(n.label.clone()))
//     }

//     fn node_shape(&'a self, _: &Nd) -> Option<LabelText<'a>> {
//         Some(LabelText::LabelStr(Cow::Owned("record".to_string())))
//     }

//     fn rank_dir(&'a self) -> Option<RankDir> {
//         Some(RankDir::LeftRight)
//     }
// }

// impl<'a> GraphWalk<'a, Nd, Ed> for Graph {
//     fn nodes(&'a self) -> Nodes<'a, Nd> {
//         let nodes: Vec<Nd> = self.nodes.values().cloned().collect();
//         Cow::Owned(nodes)
//     }

//     fn edges(&'a self) -> Edges<'a, Ed> {
//         Cow::Borrowed(&self.edges[..])
//     }

//     fn source(&'a self, e: &Ed) -> Nd {
//         self.nodes.get(&e.0).unwrap().clone()
//     }
//     fn target(&'a self, e: &Ed) -> Nd {
//         self.nodes.get(&e.1).unwrap().clone()
//     }
// }
