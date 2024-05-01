use crate::{
    var::{Var, VarRef},
    numeric::Numeric,
    scalar::Scalar,
    tensor::Tensor,
};
use std::{
    fmt::{Debug, Formatter},
    ops::Add,
};

pub trait Op: Debug + 'static {
    fn children(&self) -> Vec<VarRef>;
    fn backward(&mut self, _to: &VarRef) {}
}

#[derive(Clone, Debug)]
pub struct NoOp {}

impl Op for NoOp {
    fn children(&self) -> Vec<VarRef> {
        vec![]
    }
}

#[derive(Clone)]
pub struct PowOp<Tn: Tensor> {
    n: Tn::T,
    from: Var<Tn>,
}

impl<T: Numeric> PowOp<Scalar<T>> {
    pub fn create_flow(from: Var<Scalar<T>>, n: T) -> Var<Scalar<T>> {
        let data = Scalar::from(from.data.val().powf(n));
        let op = PowOp { n, from };

        Var::new_from_op(data, op)
    }
}

impl<T: Numeric> Debug for PowOp<Scalar<T>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{} ^ {}", self.from.data.val(), self.n)
    }
}

impl<T: Numeric> Op for PowOp<Scalar<T>> {
    fn children(&self) -> Vec<VarRef> {
        let from = self.from.clone();
        vec![from.into()]
    }

    fn backward(&mut self, to: &VarRef) {
        let to_grad: &Scalar<T> = to.grad();
        self.from.update_grad(|grad, data| {
            grad + ((self.n * data.powf(self.n - T::one())) * to_grad.val())
        });
    }
}

#[derive(Clone)]
pub struct ReluOp<Tn: Tensor> {
    from: Var<Tn>,
}

impl<T: Numeric, Tn> Debug for ReluOp<Tn>
where
    Tn: Tensor<T = T>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "ReLU({:?})", self.from.data)
    }
}

impl<T: Numeric, Tn> ReluOp<Tn>
where
    Tn: Tensor<T = T>,
{
    pub fn create_flow(from: Var<Tn>) -> Var<Tn> {
        let mut out = from.data.deep_clone();
        out.update(|v| if v.is_sign_negative() { T::zero() } else { v });

        let op = ReluOp { from };

        Var::new_from_op(out, op)
    }
}

impl<T: Numeric, Tn> Op for ReluOp<Tn>
where
    Tn: Tensor<T = T>,
{
    fn children(&self) -> Vec<VarRef> {
        let from = self.from.clone();
        vec![from.into()]
    }

    fn backward(&mut self, to: &VarRef) {
        let to_grad: &Tn = to.grad();
        let to_data: &Tn = to.data();

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
pub struct AddOp<Tn: Tensor> {
    from: (Var<Tn>, Var<Tn>),
}

impl<T: Numeric, Tn> AddOp<Tn>
where
    Tn: Tensor<T = T> + Add<Output = Tn>,
{
    pub fn create_flow(a: Var<Tn>, b: Var<Tn>) -> Var<Tn> {
        let out = a.clone().data + b.clone().data;
        let op = AddOp { from: (a, b) };

        Var::new_from_op(out, op)
    }
}

impl<Tn: Tensor> Debug for AddOp<Tn> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "+")
    }
}

impl<Tn: Tensor> Op for AddOp<Tn> {
    fn children(&self) -> Vec<VarRef> {
        let mut out = vec![self.from.0.clone().into(), self.from.1.clone().into()];
        out.sort();

        out
    }

    fn backward(&mut self, to: &VarRef) {
        let to_grad: &Tn = to.grad();
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
pub struct MulOp<Tn: Tensor> {
    from: (Var<Tn>, Var<Tn>),
}

impl<T: Numeric> Debug for MulOp<Scalar<T>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{} * {}", self.from.0.data.val(), self.from.1.data.val())
    }
}

impl<T: Numeric> MulOp<Scalar<T>> {
    pub fn create_flow(a: Var<Scalar<T>>, b: Var<Scalar<T>>) -> Var<Scalar<T>> {
        let outval = Scalar::from(a.data.val() * b.data.val());
        let op = MulOp { from: (a, b) };

        Var::new_from_op(outval, op)
    }
}

impl<T: Numeric> Op for MulOp<Scalar<T>> {
    fn children(&self) -> Vec<VarRef> {
        let mut out = vec![self.from.0.clone().into(), self.from.1.clone().into()];
        out.sort();

        out
    }

    fn backward(&mut self, to: &VarRef) {
        let to_grad: &Scalar<T> = to.grad();

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
