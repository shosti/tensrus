use crate::{flow::Flow, numeric::Numeric, scalar::Scalar, tensor::TensorOps};
use std::fmt::Debug;

pub trait Op<T: Numeric, Tn: TensorOps<T>>: Debug {
    fn children(&self) -> Vec<Flow<T, Tn>>;
    fn backward(&self, _to_grad: &Tn, _to_data: &Tn) {}
}

#[derive(Clone, Debug)]
pub struct NoOp {}

impl<T: Numeric, Tn: TensorOps<T>> Op<T, Tn> for NoOp {
    fn children(&self) -> Vec<Flow<T, Tn>> {
        vec![]
    }
}

#[derive(Clone, Debug)]
pub struct PowOp<T: Numeric, Tn: TensorOps<T>> {
    n: T,
    from: Flow<T, Tn>,
}

impl<T: Numeric> PowOp<T, Scalar<T>> {
    pub fn create_flow(from: Flow<T, Scalar<T>>, n: T) -> Flow<T, Scalar<T>> {
        let mut data = Scalar::zeros();
        data.update(&|x: T| x.powf(n));
        let op = PowOp { n, from };

        Flow::new_from_op(data, op)
    }
}

impl<T: Numeric> Op<T, Scalar<T>> for PowOp<T, Scalar<T>> {
    fn children(&self) -> Vec<Flow<T, Scalar<T>>> {
        vec![self.from.clone()]
    }

    fn backward(&self, to_grad: &Scalar<T>, _to_data: &Scalar<T>) {
        self.from.update_grad(|grad, data| {
            grad + ((self.n * data.powf(self.n - T::one())) * to_grad.val())
        });
    }
}
