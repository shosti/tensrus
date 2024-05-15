use crate::tensor::Tensor;
use std::any::Any;
use std::cell::RefCell;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::rc::Rc;
use num::Zero;

#[derive(Debug, Clone)]
pub enum Input {
    Unary(Rc<dyn Any>),
    Binary(Rc<dyn Any>, Rc<dyn Any>),
}

impl Input {
    fn downcast_unary<Tn: Tensor>(&self) -> &Tn {
        if let Input::Unary(val) = self {
            let data: &Tn = val.downcast_ref().unwrap();
            data
        } else {
            panic!("tried to call downcast_unary on a non-unary input")
        }
    }
}

impl<Tn: Tensor> From<Rc<RefCell<Tn>>> for Input {
    fn from(t: Rc<RefCell<Tn>>) -> Self {
        Self::Unary(t)
    }
}

pub trait Op: Debug {
    type Output: Tensor;

    fn forward(&self, input: Input) -> Self::Output;
    fn backward(&self, input_grads: Input, to_data: &Self::Output, to_data: &Self::Output) -> Input;
}

#[derive(Debug)]
pub struct ReLU<Tn: Tensor> {
    _markers: PhantomData<Tn>,
}

impl<Tn: Tensor> ReLU<Tn> {
    pub fn new() -> Self {
        Self {
            _markers: PhantomData,
        }
    }
}

impl<Tn: Tensor> Op for ReLU<Tn> {
    type Output = Tn;

    fn forward(&self, input: Input) -> Self::Output {
        let data: &Self::Output = input.downcast_unary();
        data.clone().relu()
    }

    fn backward(&self, input_grads: Input, to_grad: &Self::Output, to_data: &Self::Output) -> Input {
        if let Input::Unary(input_grad) = input_grads {
            let in_typed: Rc<Self::Output> = input_grad.downcast().unwrap();
            let in_grad = Rc::into_inner(in_typed).unwrap();
            let grad = in_grad.map(|idx, val| {
                val
            });

            let res = grad.map(|idx, g|{
                let diff = if to_data[idx] > Tn::T::zero() {
                    to_grad[idx]
                } else {
                    Tn::T::zero()
                };

                g + diff
            });
            Input::Unary(Rc::new(res))
        } else {
            panic!("non-unary input grads sent to relu")
        }
    }
}
// macro_rules! create_unary_op {
//     ($name:ident <$( $generic:ident ),*> { in_type: $inty:ty, out_type: $outty:ty, forward: $forward:expr , backward: $back:expr , }) => {
//         #[derive(Debug)]
//         pub struct $name<$( $generic : Tensor ),*> {
//             _markers: ( $( ::std::marker::PhantomData<$generic> )*, ),
//         }

//         impl<$( $generic: Tensor ),*> $name<$( $generic ),*> {
//             pub fn new() -> Self {
//                 Self {}
//             }
//         }

//         impl<$( $generic: Tensor ),*> Op for $name<$( $generic ),*> {
//             type Output = $outty;

//             fn forward(&self, input: &Input) -> Self::Output {
//                 if let Input::Unary(var_ref) = input {
//                     todo!()
//                     // let var_rc = var_ref.clone().into_var::<Tn>();
//                     // let var = var_rc.borrow();
//                     // var.map($forward).into()
//                 } else {
//                     panic!("non-unary input to unary op")
//                 }
//             }

//             fn backward(&self) {
//                 todo!()
//             }
//         }
//     };
// }

// create_unary_op!(ReLU<Tn> {
//     in_type: Tn,
//     out_type: Tn,
//     forward: |t| t.clone().relu(),
//     backward: |to_val, to_grad, from_val, from_grad| {
//         let diff = if to_data > 0 { to_grad } else { 0 };

//         from_grad + diff
//     },
// });
