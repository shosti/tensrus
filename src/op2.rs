use crate::{
    numeric::Numeric,
    tensor::{BasicTensor, Tensor},
};
use num::Zero;
use std::fmt::Debug;
use std::marker::PhantomData;

#[derive(Debug)]
pub enum ForwardInput<'a, T: Numeric> {
    Unary(&'a Box<dyn BasicTensor<T>>),
    Binary(&'a Box<dyn BasicTensor<T>>, &'a Box<dyn BasicTensor<T>>),
}

impl<'a, T: Numeric> ForwardInput<'a, T> {
    pub fn unary(&'a self) -> &'a Box<dyn BasicTensor<T>> {
        if let Self::Unary(t) = self {
            t
        } else {
            panic!("non-unary input")
        }
    }

    pub fn binary(&'a self) -> (&'a Box<dyn BasicTensor<T>>, &'a Box<dyn BasicTensor<T>>) {
        if let Self::Binary(t1, t2) = self {
            (t1, t2)
        } else {
            panic!("non-binary input")
        }
    }
}

#[derive(Debug)]
pub enum BackwardOutput<T: Numeric> {
    Unary(Box<dyn BasicTensor<T>>),
    Binary(Box<dyn BasicTensor<T>>, Box<dyn BasicTensor<T>>),
}

impl<T: Numeric> BackwardOutput<T> {
    pub fn unary(self) -> Box<dyn BasicTensor<T>> {
        if let Self::Unary(out) = self {
            out
        } else {
            panic!("non-unary output")
        }
    }

    pub fn binary(self) -> (Box<dyn BasicTensor<T>>, Box<dyn BasicTensor<T>>) {
        if let Self::Binary(out1, out2) = self {
            (out1, out2)
        } else {
            panic!("non-binary output")
        }
    }
}

pub trait Op<T: Numeric>: Debug {
    fn forward(&self, input: ForwardInput<T>) -> Box<dyn BasicTensor<T>>;
    fn backward<'a>(
        &self,
        in_grad: BackwardOutput<T>,
        out_data: &'a Box<dyn BasicTensor<T>>,
        out_grad: &'a Box<dyn BasicTensor<T>>,
    ) -> BackwardOutput<T>;
}

#[derive(Debug)]
pub struct ReLU<Tn: Tensor> {
    _markers: PhantomData<Tn>,
}

impl<Tn: Tensor> ReLU<Tn> {
    pub fn new() -> Box<Self> {
        Box::new(Self {
            _markers: PhantomData,
        })
    }
}

impl<Tn: Tensor> Op<Tn::T> for ReLU<Tn> {
    fn forward(&self, inputs: ForwardInput<Tn::T>) -> Box<dyn BasicTensor<Tn::T>> {
        let input = inputs.unary();
        let out = Tn::from_basic(input.as_ref()).relu();
        Box::new(out)
    }
    fn backward<'a>(
        &self,
        in_grads: BackwardOutput<Tn::T>,
        out_data: &'a Box<dyn BasicTensor<Tn::T>>,
        out_grad: &'a Box<dyn BasicTensor<Tn::T>>,
    ) -> BackwardOutput<Tn::T> {
        let in_grad_untyped = in_grads.unary();
        let in_grad = Tn::from_basic_boxed(in_grad_untyped);
        let updated_grad = in_grad.map(|idx, in_grad| {
            let diff = if out_data[idx.as_ref()] > Tn::T::zero() {
                out_grad[idx.as_ref()]
            } else {
                Tn::T::zero()
            };

            in_grad + diff
        });
        BackwardOutput::Unary(Box::new(updated_grad))
    }
}

#[derive(Debug)]
pub struct AddOp<Tn: Tensor> {
    _markers: PhantomData<Tn>,
}

impl<Tn: Tensor> AddOp<Tn> {
    pub fn new() -> Box<Self> {
        Box::new(Self {
            _markers: PhantomData,
        })
    }
}

impl<Tn: Tensor> Op<Tn::T> for AddOp<Tn> {
    fn forward(&self, inputs: ForwardInput<Tn::T>) -> Box<dyn BasicTensor<Tn::T>> {
        let (a, b) = inputs.binary();
        let out = Tn::from_basic(a.as_ref()) + Tn::ref_from_basic(b.as_ref());
        Box::new(out)
    }
    fn backward<'a>(
        &self,
        in_grads: BackwardOutput<Tn::T>,
        _out_data: &'a Box<dyn BasicTensor<Tn::T>>,
        out_grad: &'a Box<dyn BasicTensor<Tn::T>>,
    ) -> BackwardOutput<Tn::T> {
        let (a_grad_basic, b_grad_basic) = in_grads.binary();
        let a_grad = Tn::from_basic_boxed(a_grad_basic);
        let b_grad = Tn::from_basic_boxed(b_grad_basic);

        let a_grad_updated = a_grad.map(|idx, in_grad| in_grad + out_grad[idx.as_ref()]);
        let b_grad_updated = b_grad.map(|idx, in_grad| in_grad + out_grad[idx.as_ref()]);
        BackwardOutput::Binary(Box::new(a_grad_updated), Box::new(b_grad_updated))
    }
}
