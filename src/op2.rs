use crate::{
    numeric::Numeric,
    tensor::{BasicTensor, Tensor},
};
// use std::any::Any;
// use std::cell::RefCell;
use std::fmt::Debug;
use std::marker::PhantomData;
// use std::rc::Rc;
use num::Zero;

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
    fn forward(&self, input: ForwardInput<Tn::T>) -> Box<dyn BasicTensor<Tn::T>> {
        if let ForwardInput::Unary(input) = input {
            let out = Tn::from_basic(input.as_ref()).relu();
            Box::new(out)
        } else {
            panic!("non-unary input to ReLU")
        }
    }
    fn backward<'a>(
        &self,
        out_data: &'a Box<dyn BasicTensor<Tn::T>>,
        out_grad: &'a Box<dyn BasicTensor<Tn::T>>,
    ) -> BackwardOutput<Tn::T> {
        let t: Tn = Tn::from_basic(out_grad.as_ref());
        let in_grad = t.map(|idx, out_grad_val| {
            if out_data[idx.as_ref()] > Tn::T::zero() {
                out_grad_val
            } else {
                Tn::T::zero()
            }
        });
        BackwardOutput::Unary(Box::new(in_grad))
    }
}

// // macro_rules! create_unary_op {
// //     ($name:ident <$( $generic:ident ),*> { in_type: $inty:ty, out_type: $outty:ty, forward: $forward:expr , backward: $back:expr , }) => {
// //         #[derive(Debug)]
// //         pub struct $name<$( $generic : Tensor ),*> {
// //             _markers: ( $( ::std::marker::PhantomData<$generic> )*, ),
// //         }

// //         impl<$( $generic: Tensor ),*> $name<$( $generic ),*> {
// //             pub fn new() -> Self {
// //                 Self {}
// //             }
// //         }

// //         impl<$( $generic: Tensor ),*> Op for $name<$( $generic ),*> {
// //             type Output = $outty;

// //             fn forward(&self, input: &Input) -> Self::Output {
// //                 if let Input::Unary(var_ref) = input {
// //                     todo!()
// //                     // let var_rc = var_ref.clone().into_var::<Tn>();
// //                     // let var = var_rc.borrow();
// //                     // var.map($forward).into()
// //                 } else {
// //                     panic!("non-unary input to unary op")
// //                 }
// //             }

// //             fn backward(&self) {
// //                 todo!()
// //             }
// //         }
// //     };
// // }

// // create_unary_op!(ReLU<Tn> {
// //     in_type: Tn,
// //     out_type: Tn,
// //     forward: |t| t.clone().relu(),
// //     backward: |to_val, to_grad, from_val, from_grad| {
// //         let diff = if to_data > 0 { to_grad } else { 0 };

// //         from_grad + diff
// //     },
// // });
