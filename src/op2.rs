use crate::flow::Id;
use crate::tensor::Tensor;
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub enum Input {
    Unary(Id),
    Binary(Id, Id),
}

pub trait Op: Debug {
    type Output: Tensor;

    fn forward(&self, input: &Input) -> Self::Output;
    fn backward(&self);
}

macro_rules! create_unary_op {
    ($name:ident <$( $generic:ident ),*> { in_type: $inty:ty, out_type: $outty:ty, forward: $forward:expr , backward: $back:expr , }) => {
        #[derive(Debug)]
        struct $name<$( $generic : Tensor ),*> {
            _markers: ( $( ::std::marker::PhantomData<$generic> )*, ),
        }

        impl<$( $generic: Tensor ),*> Op for $name<$( $generic ),*> {
            type Output = $outty;

            fn forward(&self, input: &Input) -> Self::Output {
                if let Input::Unary(var_ref) = input {
                    todo!()
                    // let var_rc = var_ref.clone().into_var::<Tn>();
                    // let var = var_rc.borrow();
                    // var.map($forward).into()
                } else {
                    panic!("non-unary input to unary op")
                }
            }

            fn backward(&self) {
                todo!()
            }
        }
    };
}

create_unary_op!(ReLU<Tn> {
    in_type: Tn,
    out_type: Tn,
    forward: |t| t.clone().relu(),
    backward: |to_val, to_grad, from_val, from_grad| {
        let diff = if to_data > 0 { to_grad } else { 0 };

        from_grad + diff
    },
});
