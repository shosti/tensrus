use crate::{flow::Flow, numeric::Numeric, scalar::Scalar, tensor::TensorOps};
use std::fmt::{Formatter, Debug};

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

#[derive(Clone)]
pub struct PowOp<T: Numeric, Tn: TensorOps<T>> {
    n: T,
    from: Flow<T, Tn>,
}

impl<T: Numeric> PowOp<T, Scalar<T>> {
    pub fn create_flow(from: Flow<T, Scalar<T>>, n: T) -> Flow<T, Scalar<T>> {
        let data = Scalar::from(from.val().powf(n));
        let op = PowOp { n, from };

        Flow::new_from_op(data, op)
    }
}

impl<T: Numeric> Debug for PowOp<T, Scalar<T>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{} ^ {}", self.from.val(), self.n)
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

#[derive(Clone)]
pub struct ReluOp<T: Numeric, Tn: TensorOps<T>> {
    from: Flow<T, Tn>,
}

impl<T: Numeric> Debug for ReluOp<T, Scalar<T>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "ReLU({})", self.from.val())
    }
}

impl<T: Numeric> ReluOp<T, Scalar<T>> {
    pub fn create_flow(from: Flow<T, Scalar<T>>) -> Flow<T, Scalar<T>> {
        let data = from.val();
        let outval = if data.is_sign_negative() {
            Scalar::from(T::zero())
        } else {
            Scalar::from(data)
        };
        let op = ReluOp { from };

        Flow::new_from_op(outval, op)
    }
}

impl<T: Numeric> Op<T, Scalar<T>> for ReluOp<T, Scalar<T>> {
    fn children(&self) -> Vec<Flow<T, Scalar<T>>> {
        vec![self.from.clone()]
    }

    fn backward(&self, to_grad: &Scalar<T>, to_data: &Scalar<T>) {
        self.from.update_grad(|grad, _data| {
            let diff = if to_data.val().is_sign_positive() && !to_data.val().is_zero() {
                to_grad.val()
            } else {
                T::zero()
            };

            grad + diff
        });
    }
}

#[derive(Clone)]
pub struct AddOp<T: Numeric, Tn: TensorOps<T>> {
    from: (Flow<T, Tn>, Flow<T, Tn>),
}

impl<T: Numeric> AddOp<T, Scalar<T>> {
    pub fn create_flow(a: Flow<T, Scalar<T>>, b: Flow<T, Scalar<T>>) -> Flow<T, Scalar<T>> {
        let outval = Scalar::from(a.val() + b.val());
        let op = AddOp { from: (a, b) };

        Flow::new_from_op(outval, op)
    }
}

impl<T: Numeric> Debug for AddOp<T, Scalar<T>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{} + {}", self.from.0.val(), self.from.1.val())
    }
}

impl<T: Numeric> Op<T, Scalar<T>> for AddOp<T, Scalar<T>> {
    fn children(&self) -> Vec<Flow<T, Scalar<T>>> {
        let mut out = vec![self.from.0.clone(), self.from.1.clone()];
        out.sort();

        out
    }

    fn backward(&self, to_grad: &Scalar<T>, _to_data: &Scalar<T>) {
        self.from.0.update_grad(|grad, _data| {
            grad + to_grad.val()
        });
        self.from.1.update_grad(|grad, _data| {
            grad + to_grad.val()
        });
    }
}

#[derive(Clone)]
pub struct MulOp<T: Numeric, Tn: TensorOps<T>> {
    from: (Flow<T, Tn>, Flow<T, Tn>),
}

impl<T: Numeric> Debug for MulOp<T, Scalar<T>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{} * {}", self.from.0.val(), self.from.1.val())
    }
}

impl<T: Numeric> MulOp<T, Scalar<T>> {
    pub fn create_flow(a: Flow<T, Scalar<T>>, b: Flow<T, Scalar<T>>) -> Flow<T, Scalar<T>> {
        let outval = Scalar::from(a.val() * b.val());
        let op = MulOp { from: (a, b) };

        Flow::new_from_op(outval, op)
    }
}

impl<T: Numeric> Op<T, Scalar<T>> for MulOp<T, Scalar<T>> {
    fn children(&self) -> Vec<Flow<T, Scalar<T>>> {
        let mut out = vec![self.from.0.clone(), self.from.1.clone()];
        out.sort();

        out
    }

    fn backward(&self, _to_grad: &Scalar<T>, _to_data: &Scalar<T>) {
        let a_data = self.from.0.val();
        let b_data = self.from.0.val();

        self.from.0.update_grad(|grad, _data| {
            grad + b_data * grad
        });
        self.from.1.update_grad(|grad, _data| {
            grad + a_data * grad
        });
    }
}
