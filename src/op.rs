use crate::{
    flow::Flow,
    numeric::Numeric,
    tensor::{TensorOps},
};
use std::fmt::{Debug, Formatter};

#[derive(Debug)]
pub enum Op<T: Numeric, Tn: TensorOps<T>> {
    None,
    Unary(UnaryOp<T, Tn>),
    Binary(BinaryOp<T, Tn>),
}

pub struct UnaryOp<T: Numeric, Tn: TensorOps<T>> {
    op: String,
    child: Flow<T, Tn>,
    _f: Box<dyn FnMut(Flow<T, Tn>, T)>,
}

impl<T: Numeric, Tn: TensorOps<T>> Debug for UnaryOp<T, Tn> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{}({:?})", self.op, self.child)
    }
}

pub struct BinaryOp<T: Numeric, Tn: TensorOps<T>> {
    op: String,
    children: (Flow<T, Tn>, Flow<T, Tn>),
    _f: Box<dyn FnMut(Flow<T, Tn>, Flow<T, Tn>, T)>,
}

impl<T: Numeric, Tn: TensorOps<T>> Debug for BinaryOp<T, Tn> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        let (ch1, ch2) = &self.children;
        write!(f, "{}({:?}, {:?})", self.op, ch1, ch2)
    }
}

impl<T: Numeric, Tn: TensorOps<T>> Op<T, Tn> {
    pub fn children(&self) -> Vec<Flow<T, Tn>> {
        match self {
            Op::None => vec![],
            Op::Unary(UnaryOp { child, .. }) => vec![child.clone()],
            Op::Binary(BinaryOp {
                children: (ch1, ch2),
                ..
            }) => {
                let mut res = vec![ch1.clone(), ch2.clone()];
                res.sort();
                res
            }
        }
    }
}
