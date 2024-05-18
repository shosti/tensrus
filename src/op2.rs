use crate::{
    numeric::Numeric,
    tensor::{BasicTensor},
};
use num::Zero;
use std::fmt::Debug;

#[derive(Debug)]
pub enum ForwardInput<'a, T: Numeric> {
    Unary(&'a Box<dyn BasicTensor<T>>),
    Binary(&'a Box<dyn BasicTensor<T>>, &'a Box<dyn BasicTensor<T>>),
}

#[derive(Debug)]
pub enum BackwardOutput<T: Numeric> {
    Unary(Box<dyn BasicTensor<T>>),
    Binary(Box<dyn BasicTensor<T>>, Box<dyn BasicTensor<T>>),
}

pub trait Op<T: Numeric>: Debug {
    fn forward(&self, input: ForwardInput<T>) -> Box<dyn BasicTensor<T>>;
    fn backward<'a>(
        &self,
        out_data: &'a Box<dyn BasicTensor<T>>,
        out_grad: &'a Box<dyn BasicTensor<T>>,
    ) -> BackwardOutput<T>;
}

macro_rules! create_unary_op {
    ($name:ident { forward: $forward:expr , backward: $backward:expr , }) => {
        #[derive(Debug)]
        pub struct $name<Tn: crate::tensor::Tensor> {
            _marker: ::std::marker::PhantomData<Tn>,
        }

        impl<Tn: crate::tensor::Tensor> $name<Tn> {
            pub fn new() -> Box<Self> {
                Box::new(Self {
                    _marker: ::std::marker::PhantomData,
                })
            }
        }

        impl<Tn: crate::tensor::Tensor> Op<Tn::T> for $name<Tn> {
            fn forward(&self, input: ForwardInput<Tn::T>) -> Box<dyn BasicTensor<Tn::T>> {
                if let ForwardInput::Unary(input) = input {
                    let input: Tn = Tn::from_basic(input.as_ref());
                    let forward = $forward(input);
                    Box::new(forward)
                } else {
                    panic!("non-unary input to ReLU")
                }
            }
            fn backward<'a>(
                &self,
                out_data_tn: &'a Box<dyn BasicTensor<Tn::T>>,
                out_grad_tn: &'a Box<dyn BasicTensor<Tn::T>>,
            ) -> BackwardOutput<Tn::T> {
                let t: Tn = Tn::from_basic(out_grad_tn.as_ref());
                let in_grad = t.map(|idx, out_grad| {
                    let out_data = out_data_tn[idx.as_ref()];
                    $backward(out_grad, out_data)
                });
                BackwardOutput::Unary(Box::new(in_grad))
            }
        }
    };
}

create_unary_op!(ReLU {
    forward: |input: Tn| input.relu(),
    backward: |out_grad, out_data| if out_data > Tn::T::zero() {
        out_grad
    } else {
        Tn::T::zero()
    },
});
