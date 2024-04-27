use crate::{flow::Flow, numeric::Numeric, tensor::TensorOps};
use std::fmt::Debug;

pub trait Op<T: Numeric, Tn: TensorOps<T>>: Debug {
    fn children(&self) -> Vec<Flow<T, Tn>>;
}

#[derive(Clone, Debug)]
pub struct NoOp {}

impl<T: Numeric, Tn: TensorOps<T>> Op<T, Tn> for NoOp {
    fn children(&self) -> Vec<Flow<T, Tn>> {
        vec![]
    }
}
