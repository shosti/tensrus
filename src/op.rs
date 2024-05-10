use crate::{
    matrix::{matrix_shape, Matrix},
    numeric::Numeric,
    scalar::Scalar,
    tensor::{num_elems, Tensor},
    var::{Var, VarRef},
};
use num::{Float, Zero};
use std::fmt::{Debug, Formatter};

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

impl<Tn> Debug for ReluOp<Tn>
where
    Tn: Tensor,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "ReLU({:?})", self.from.data)
    }
}

impl<Tn> ReluOp<Tn>
where
    Tn: Tensor,
{
    pub fn create_flow(from: Var<Tn>) -> Var<Tn> {
        let out = from.data.clone().map(|v, _| {
            if v.is_sign_negative() {
                Tn::T::zero()
            } else {
                v
            }
        });

        let op = ReluOp { from };

        Var::new_from_op(out, op)
    }
}

impl<Tn> Op for ReluOp<Tn>
where
    Tn: Tensor,
{
    fn children(&self) -> Vec<VarRef> {
        let from = self.from.clone();
        vec![from.into()]
    }

    fn backward(&mut self, _to: &VarRef) {
        // let to_grad: &Tn = to.grad();
        // let to_data: &Tn = to.data();

        // self.from
        //     .grad
        //     .update_zip2(to_grad, to_data, |from_grad, to_grad, to_data| {
        //         let diff = if to_data.is_sign_positive() && !to_data.is_zero() {
        //             to_grad
        //         } else {
        //             T::zero()
        //         };

        //         from_grad + diff
        //     })
    }
}

#[derive(Clone)]
pub struct AddOp<Tn: Tensor> {
    from: (Var<Tn>, Var<Tn>),
}

impl<Tn: Tensor> AddOp<Tn> {
    pub fn create_flow(a: &Var<Tn>, b: &Var<Tn>) -> Var<Tn> {
        let out = a.data.clone() + &b.data;
        let op = AddOp {
            from: (a.clone(), b.clone()),
        };

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

    fn backward(&mut self, _to: &VarRef) {
        // let to_grad: &Tn = to.grad();
        // self.from
        //     .0
        //     .grad
        //     .update_zip(to_grad, |from_grad, to_grad| from_grad + to_grad);
        // self.from
        //     .1
        //     .grad
        //     .update_zip(to_grad, |from_grad, to_grad| from_grad + to_grad);
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

#[derive(Clone)]
pub struct MatMulOp<T: Numeric, const M: usize, const N: usize, const P: usize>
where
    [(); num_elems(2, matrix_shape(M, N))]:,
    [(); num_elems(2, matrix_shape(N, P))]:,
{
    from: (Var<Matrix<T, M, N>>, Var<Matrix<T, N, P>>),
}

impl<T: Numeric, const M: usize, const N: usize, const P: usize> MatMulOp<T, M, N, P>
where
    [(); num_elems(2, matrix_shape(M, N))]:,
    [(); num_elems(2, matrix_shape(N, P))]:,
    [(); num_elems(2, matrix_shape(M, P))]:,
{
    pub fn create_flow(a: Var<Matrix<T, M, N>>, b: Var<Matrix<T, N, P>>) -> Var<Matrix<T, M, P>> {
        let outval = &a.data * &b.data;

        let op = Self { from: (a, b) };

        Var::new_from_op(outval, op)
    }
}

impl<T: Numeric, const M: usize, const N: usize, const P: usize> Debug for MatMulOp<T, M, N, P>
where
    [(); num_elems(2, matrix_shape(M, N))]:,
    [(); num_elems(2, matrix_shape(N, P))]:,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "*")
    }
}

impl<T: Numeric, const M: usize, const N: usize, const P: usize> Op for MatMulOp<T, M, N, P>
where
    [(); num_elems(2, matrix_shape(M, N))]:,
    [(); num_elems(2, matrix_shape(N, P))]:,
{
    fn children(&self) -> Vec<VarRef> {
        let mut out = vec![self.from.0.clone().into(), self.from.1.clone().into()];
        out.sort();

        out
    }

    fn backward(&mut self, _to: &VarRef) {
        // let to_grad: &Matrix<T, M, P> = to.grad();

        // let a_data = self.from.0.data;
        // let b_data = self.from.1.data;

        // self.from
        //     .0
        //     .update_grad(|grad, _data| grad + b_data * to_grad.val());
        // self.from
        //     .1
        //     .update_grad(|grad, _data| grad + a_data * to_grad.val());
    }
}
