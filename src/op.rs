use crate::{
    flow::Flow,
    numeric::Numeric,
    tensor::{Tensor, TensorOps, TensorShape},
};
use std::{
    fmt::{Debug, Formatter},
};

#[derive(Debug)]
pub enum Op<T: Numeric, const R: usize, const S: TensorShape, Tn>
where
    Tn: Tensor<T, R, S> + TensorOps<T>,
{
    None,
    Unary(UnaryOp<T, R, S, Tn>),
    Binary(BinaryOp<T, R, S, Tn>),
}

pub struct UnaryOp<T: Numeric, const R: usize, const S: TensorShape, Tn>
where
    Tn: Tensor<T, R, S> + TensorOps<T>,
{
    op: String,
    child: Flow<T, R, S, Tn>,
    _f: Box<dyn FnMut(Flow<T, R, S, Tn>, T)>,
}

impl<T: Numeric, const R: usize, const S: TensorShape, Tn> Debug for UnaryOp<T, R, S, Tn>
where
    Tn: Tensor<T, R, S> + TensorOps<T> + Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{}({:?})", self.op, self.child)
    }
}

pub struct BinaryOp<T: Numeric, const R: usize, const S: TensorShape, Tn>
where
    Tn: Tensor<T, R, S> + TensorOps<T>,
{
    op: String,
    children: (Flow<T, R, S, Tn>, Flow<T, R, S, Tn>),
    _f: Box<dyn FnMut(Flow<T, R, S, Tn>, Flow<T, R, S, Tn>, T)>,
}

impl<T: Numeric, const R: usize, const S: TensorShape, Tn> Debug for BinaryOp<T, R, S, Tn>
where
    Tn: Tensor<T, R, S> + TensorOps<T> + Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        let (ch1, ch2) = &self.children;
        write!(f, "{}({:?}, {:?})", self.op, ch1, ch2)
    }
}

// impl<T: Numeric, const R: usize, const S: TensorShape, Tn> Op<T, R, S, Tn> where
//     Tn: Tensor<T, R, S> + TensorOps<T>
// {
//     pub fn children() -> Vec<
// }
