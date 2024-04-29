use crate::{
    flow::{Flow, FlowRef},
    numeric::Numeric,
    scalar::Scalar,
    tensor::{BasicTensor, Tensor},
};
use std::{
    fmt::{Debug, Formatter},
    ops::Add,
};

pub trait Op<T: Numeric>: Debug + 'static {
    fn children(&self) -> Vec<FlowRef<T>>;
    fn backward(&mut self, _to_grad: &dyn BasicTensor<T>, _to_data: &dyn BasicTensor<T>) {}
}

#[derive(Clone, Debug)]
pub struct NoOp {}

impl<T: Numeric> Op<T> for NoOp {
    fn children(&self) -> Vec<FlowRef<T>> {
        vec![]
    }
}

#[derive(Clone)]
pub struct PowOp<T: Numeric, Tn: Tensor<T>> {
    n: T,
    from: Flow<T, Tn>,
}

impl<T: Numeric> PowOp<T, Scalar<T>> {
    pub fn create_flow(from: Flow<T, Scalar<T>>, n: T) -> Flow<T, Scalar<T>> {
        let data = Scalar::from(from.data.val().powf(n));
        let op = PowOp { n, from };

        Flow::new_from_op(data, op)
    }
}

impl<T: Numeric> Debug for PowOp<T, Scalar<T>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{} ^ {}", self.from.data.val(), self.n)
    }
}

impl<T: Numeric> Op<T> for PowOp<T, Scalar<T>> {
    fn children(&self) -> Vec<FlowRef<T>> {
        let from = self.from.clone();
        vec![from.into()]
    }

    fn backward(&mut self, to_grad_uncast: &dyn BasicTensor<T>, _to_data: &dyn BasicTensor<T>) {
        let to_grad: &Scalar<T> = to_grad_uncast.as_any().downcast_ref().unwrap();
        self.from.update_grad(|grad, data| {
            grad + ((self.n * data.powf(self.n - T::one())) * to_grad.val())
        });
    }
}

#[derive(Clone)]
pub struct ReluOp<T: Numeric, Tn: Tensor<T>> {
    from: Flow<T, Tn>,
}

impl<T: Numeric, Tn> Debug for ReluOp<T, Tn>
where
    Tn: Tensor<T>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "ReLU({:?})", self.from.data)
    }
}

impl<T: Numeric, Tn> ReluOp<T, Tn>
where
    Tn: Tensor<T>,
{
    pub fn create_flow(from: Flow<T, Tn>) -> Flow<T, Tn> {
        let mut out = from.data.deep_clone();
        out.update(|v| if v.is_sign_negative() { T::zero() } else { v });

        let op = ReluOp { from };

        Flow::new_from_op(out, op)
    }
}

impl<T: Numeric, Tn> Op<T> for ReluOp<T, Tn>
where
    Tn: Tensor<T>,
{
    fn children(&self) -> Vec<FlowRef<T>> {
        let from = self.from.clone();
        vec![from.into()]
    }

    fn backward(
        &mut self,
        to_grad_uncast: &dyn BasicTensor<T>,
        to_data_uncast: &dyn BasicTensor<T>,
    ) {
        let to_grad: &Tn = to_grad_uncast.as_any().downcast_ref().unwrap();
        let to_data: &Tn = to_data_uncast.as_any().downcast_ref().unwrap();

        self.from
            .grad
            .update_zip2(to_grad, to_data, |from_grad, to_grad, to_data| {
                let diff = if to_data.is_sign_positive() && !to_data.is_zero() {
                    to_grad
                } else {
                    T::zero()
                };

                from_grad + diff
            })
    }
}

#[derive(Clone)]
pub struct AddOp<T: Numeric, Tn: Tensor<T>> {
    from: (Flow<T, Tn>, Flow<T, Tn>),
}

impl<T: Numeric, Tn> AddOp<T, Tn>
where
    Tn: Tensor<T> + Add<Output = Tn>,
{
    pub fn create_flow(a: Flow<T, Tn>, b: Flow<T, Tn>) -> Flow<T, Tn> {
        let out = a.clone().data + b.clone().data;
        let op = AddOp { from: (a, b) };

        Flow::new_from_op(out, op)
    }
}

impl<T: Numeric, Tn: Tensor<T>> Debug for AddOp<T, Tn> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "+")
    }
}

impl<T: Numeric, Tn: Tensor<T>> Op<T> for AddOp<T, Tn> {
    fn children(&self) -> Vec<FlowRef<T>> {
        let mut out = vec![self.from.0.clone().into(), self.from.1.clone().into()];
        out.sort();

        out
    }

    fn backward(&mut self, to_grad_uncast: &dyn BasicTensor<T>, _to_data: &dyn BasicTensor<T>) {
        let to_grad: &Tn = to_grad_uncast.as_any().downcast_ref().unwrap();
        self.from
            .0
            .grad
            .update_zip(to_grad, |from_grad, to_grad| from_grad + to_grad);
        self.from
            .1
            .grad
            .update_zip(to_grad, |from_grad, to_grad| from_grad + to_grad);
    }
}

#[derive(Clone)]
pub struct MulOp<T: Numeric, Tn: Tensor<T>> {
    from: (Flow<T, Tn>, Flow<T, Tn>),
}

impl<T: Numeric> Debug for MulOp<T, Scalar<T>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{} * {}", self.from.0.data.val(), self.from.1.data.val())
    }
}

impl<T: Numeric> MulOp<T, Scalar<T>> {
    pub fn create_flow(a: Flow<T, Scalar<T>>, b: Flow<T, Scalar<T>>) -> Flow<T, Scalar<T>> {
        let outval = Scalar::from(a.data.val() * b.data.val());
        let op = MulOp { from: (a, b) };

        Flow::new_from_op(outval, op)
    }
}

impl<T: Numeric> Op<T> for MulOp<T, Scalar<T>> {
    fn children(&self) -> Vec<FlowRef<T>> {
        let mut out = vec![self.from.0.clone().into(), self.from.1.clone().into()];
        out.sort();

        out
    }

    fn backward(&mut self, to_grad_uncast: &dyn BasicTensor<T>, _to_data: &dyn BasicTensor<T>) {
        let to_grad: &Scalar<T> = to_grad_uncast.as_any().downcast_ref().unwrap();

        let a_data = self.from.0.data.val();
        let b_data = self.from.1.data.val();

        self.from
            .0
            .update_grad(|grad, _data| grad + b_data * to_grad.val());
        self.from
            .1
            .update_grad(|grad, _data| grad + a_data * to_grad.val());
    }
}
