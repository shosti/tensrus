use crate::{
    differentiable::{Differentiable, DifferentiableTensor},
    dyn_tensor::{DynTensor, FromDynTensor},
    matrix::Matrix,
    scalar::Scalar,
    shape::{Broadcastable, Reducible, Transposable},
    tensor::Tensor,
    vector::Vector,
};
use num::{Float, One, Zero};
use static_assertions::assert_obj_safe;
use std::{fmt::Debug, marker::PhantomData, ops::Add};

pub trait Op: Debug {
    fn forward(&self, input: ForwardInput) -> Box<dyn DynTensor>;
    fn backward(&self, args: BackwardArgs) -> BackwardOutput;
}

assert_obj_safe!(Op);

#[derive(Debug)]
pub enum ForwardInput<'a> {
    Unary(&'a dyn DynTensor),
    Binary(&'a dyn DynTensor, &'a dyn DynTensor),
}

#[derive(Debug)]
pub enum BackwardArgs<'a> {
    Unary {
        in_grad: Box<dyn DynTensor>,
        in_data: &'a dyn DynTensor,
        out_grad: &'a dyn DynTensor,
        out_data: &'a dyn DynTensor,
    },
    Binary {
        in_grad: (Box<dyn DynTensor>, Box<dyn DynTensor>),
        in_data: (&'a dyn DynTensor, &'a dyn DynTensor),
        out_grad: &'a dyn DynTensor,
        out_data: &'a dyn DynTensor,
    },
}

#[derive(Debug)]
pub enum BackwardOutput {
    Unary(Box<dyn DynTensor>),
    Binary(Box<dyn DynTensor>, Box<dyn DynTensor>),
}

impl BackwardOutput {
    pub fn unary(self) -> Box<dyn DynTensor> {
        if let Self::Unary(out) = self {
            out
        } else {
            unreachable!()
        }
    }

    pub fn binary(self) -> (Box<dyn DynTensor>, Box<dyn DynTensor>) {
        if let Self::Binary(out1, out2) = self {
            (out1, out2)
        } else {
            unreachable!()
        }
    }
}

impl<'a> ForwardInput<'a> {
    pub fn unary(&'a self) -> &'a dyn DynTensor {
        if let Self::Unary(t) = self {
            *t
        } else {
            unreachable!()
        }
    }

    pub fn binary(&'a self) -> (&'a dyn DynTensor, &'a dyn DynTensor) {
        if let Self::Binary(t1, t2) = self {
            (*t1, *t2)
        } else {
            unreachable!()
        }
    }
}

macro_rules! unary_op {
    ($name:ident < $( $generic:ident : $subtype:ident ),* > {
        args: ( $( $arg:ident : $argty:ty ),* ),
        where_clauses: ( $( $whereclauses:tt )* ),
        const_params: ( $( $constparam:ident : $constparamty:ty ),* ),
        in_type: $inty:ty,
        out_type: $outty:ty,
        numeric_type: $numty:ty,
        forward: $forward:expr,
        backward: $backward:expr,
    }) => {

        #[derive(Debug)]
        pub struct $name< $( $generic : $subtype ),* $(, const $constparam : $constparamty )* >
        where
            $numty: Differentiable,
            $( $whereclauses )*
        {
            _markers: ( $( std::marker::PhantomData< $generic > ,)* ),
            args: ( $( $argty ,)* ),
        }

        impl< $( $generic : $subtype ),* $(, const $constparam : $constparamty )* > $name< $( $generic ),* $(, $constparam )* >
        where
            $numty: Differentiable,
            $( $whereclauses )*
        {
            pub fn new( $( $arg : $argty ),* ) -> Box<Self> {
                Box::new(Self {
                    _markers: ( $(PhantomData::< $generic >,)* ),
                    args: ( $( $arg ,)* ),
                })
            }
        }

        impl< $( $generic : $subtype ),* $(, const $constparam : $constparamty )* > Op for $name< $( $generic ),* $(, $constparam )* >
        where
            $numty: Differentiable,
            $( $whereclauses )*
        {
            fn forward(&self, inputs: ForwardInput) -> Box<dyn DynTensor> {
                let input = <$inty>::ref_from_dyn(inputs.unary());
                let out: $outty = ($forward)(input, self.args);
                Box::new(out)
            }
            fn backward<'a>(&self, args: BackwardArgs) -> BackwardOutput {
                if let BackwardArgs::Unary {
                    in_grad: in_grad_basic,
                    in_data: in_data_basic,
                    out_grad: out_grad_basic,
                    out_data: out_data_basic,
                } = args
                {
                    let in_grad = *<$inty>::from_dyn(in_grad_basic);
                    let in_data = <$inty>::ref_from_dyn(in_data_basic);
                    let out_grad = <$outty>::ref_from_dyn(out_grad_basic);
                    let out_data = <$outty>::ref_from_dyn(out_data_basic);
                    let args = UnaryBackwardArgs::<$inty, $outty, ( $( $argty ,)* )> {
                        in_data,
                        out_grad,
                        out_data,
                        args: &self.args,
                    };

                    let in_grad_updated: $inty = ($backward)(in_grad, args);

                    BackwardOutput::Unary(Box::new(in_grad_updated))
                } else {
                    unreachable!()
                }
            }
        }
    };
}

macro_rules! binary_op {
    ($name:ident < $( $generic:ident : $subtype:ident ),* > {
        args: ( $( $arg:ident : $argty:ty ),* ),
        where_clauses: ( $( $whereclauses:tt )* ),
        const_params: ( $( $constparam:ident : $constparamty:ty ),* ),
        in_type_1: $inty1:ty,
        in_type_2: $inty2:ty,
        out_type: $outty:ty,
        numeric_type: $numty:ty,
        forward: $forward:expr,
        backward_1: $backward1:expr,
        backward_2: $backward2:expr,
    }) => {

        #[derive(Debug)]
        pub struct $name< $( $generic : $subtype ),* $(, const $constparam : $constparamty )* >
        where
            $numty: Differentiable,
            $( $whereclauses )*
        {
            _markers: ( $( std::marker::PhantomData< $generic > ,)* ),
            args: ( $( $argty ,)* ),
        }

        impl< $( $generic : $subtype ),* $(, const $constparam : $constparamty )* > $name< $( $generic ),* $(, $constparam )* >
        where
            $numty: Differentiable,
            $( $whereclauses )*
        {
            pub fn new( $( $arg : $argty ),* ) -> Box<Self> {
                Box::new(Self {
                    _markers: ( $(PhantomData::< $generic >,)* ),
                    args: ( $( $arg ,)* ),
                })
            }
        }

        impl< $( $generic : $subtype ),* $(, const $constparam : $constparamty )* > Op for $name< $( $generic ),* $(, $constparam )* >
        where
            $numty: Differentiable,
            $( $whereclauses )*
        {
            fn forward(&self, inputs: ForwardInput) -> Box<dyn DynTensor> {
                let (input_1_basic, input_2_basic) = inputs.binary();
                let input_1 = <$inty1>::ref_from_dyn(input_1_basic);
                let input_2 = <$inty2>::ref_from_dyn(input_2_basic);
                let out: $outty = ($forward)(input_1, input_2, self.args);

                Box::new(out)
            }

            fn backward<'a>(&self, args: BackwardArgs) -> BackwardOutput {
                if let BackwardArgs::Binary {
                    in_grad: (in_grad_basic_1, in_grad_basic_2),
                    in_data: (in_data_basic_1, in_data_basic_2),
                    out_grad: out_grad_basic,
                    out_data: out_data_basic,
                } = args
                {
                    let in_grad_1 = *<$inty1>::from_dyn(in_grad_basic_1);
                    let in_grad_2 = *<$inty2>::from_dyn(in_grad_basic_2);
                    let in_data_1 = <$inty1>::ref_from_dyn(in_data_basic_1);
                    let in_data_2 = <$inty2>::ref_from_dyn(in_data_basic_2);
                    let out_grad = <$outty>::ref_from_dyn(out_grad_basic);
                    let out_data = <$outty>::ref_from_dyn(out_data_basic);

                    let args_1 = BinaryBackwardArgs {
                        _self_in_data: in_data_1,
                        other_in_data: in_data_2,
                        out_grad,
                        _out_data: out_data,
                        _args: &self.args,
                    };
                    let args_2 = BinaryBackwardArgs {
                        _self_in_data: in_data_2,
                        other_in_data: in_data_1,
                        out_grad,
                        _out_data: out_data,
                        _args: &self.args,
                    };
                    let in_grad_1_updated: $inty1 = ($backward1)(in_grad_1, args_1);
                    let in_grad_2_updated: $inty2 = ($backward2)(in_grad_2, args_2);

                    BackwardOutput::Binary(Box::new(in_grad_1_updated), Box::new(in_grad_2_updated))
                } else {
                    unreachable!()
                }
            }
        }
    };
}

struct UnaryBackwardArgs<'a, InTn, OutTn, Args> {
    in_data: &'a InTn,
    out_grad: &'a OutTn,
    out_data: &'a OutTn,
    args: &'a Args,
}

struct BinaryBackwardArgs<'a, SelfInTn, OtherInTn, OutTn, Args> {
    _self_in_data: &'a SelfInTn,
    other_in_data: &'a OtherInTn,
    out_grad: &'a OutTn,
    _out_data: &'a OutTn,
    _args: &'a Args,
}

unary_op!(ReLUOp<Tn: DifferentiableTensor> {
    args: (),
    where_clauses: (),
    const_params: (),
    in_type: Tn,
    out_type: Tn,
    numeric_type: Tn::T,
    forward: |input: &Tn, _args: ()| input.clone().relu(),
    backward: (|in_grad: Tn, args: UnaryBackwardArgs<Tn, Tn, _>|
               in_grad.map(|idx, in_grad| {
                   let diff = if args.out_data[idx] > Tn::T::zero() {
                       args.out_grad[idx]
                   } else {
                       Tn::T::zero()
                   };

                   *in_grad + diff
               })),
});

unary_op!(ElemLnOp<Tn: DifferentiableTensor> {
    args: (),
    where_clauses: (),
    const_params: (),
    in_type: Tn,
    out_type: Tn,
    numeric_type: Tn::T,
    forward: |input: &Tn, _args: ()| input.clone().map(|_, x| x.ln()),
    backward: (|in_grad: Tn, args: UnaryBackwardArgs<Tn, Tn, _>|
               in_grad.map(|idx, in_grad| {
                   *in_grad + (Tn::T::one() / args.in_data[idx])
               })),
});

unary_op!(ElemPowOp<Tn: DifferentiableTensor> {
    args: (n: Tn::T),
    where_clauses: (),
    const_params: (),
    in_type: Tn,
    out_type: Tn,
    numeric_type: Tn::T,
    forward: |input: &Tn, (n,): (Tn::T,)| Tn::from_fn(|idx| input[idx].powf(n)),
    backward: (|in_grad: Tn, args: UnaryBackwardArgs<Tn, Tn, (Tn::T,)>|
               in_grad.map(|idx, in_grad| {
                   let out_grad_val = args.out_grad[idx];
                   let in_data_val = args.in_data[idx];
                   let n = args.args.0;

                   *in_grad + ((n * in_data_val.powf(n - Tn::T::one())) * out_grad_val)
               })),
});

unary_op!(SumOp<Tn: DifferentiableTensor> {
    args: (),
    where_clauses: (),
    const_params: (),
    in_type: Tn,
    out_type: Scalar<Tn::T>,
    numeric_type: Tn::T,
    forward: |input: &Tn, _| input.sum(),
    backward: |in_grad: Tn, _| in_grad.map(|_, v| *v + Tn::T::one()),
});

unary_op!(DimSumOp<Tn: DifferentiableTensor, Dest: DifferentiableTensor> {
    args: (),
    where_clauses: (Tn: Reducible<Dest, DIM>,
                    Dest: DifferentiableTensor<T = Tn::T, Idx = Tn::Idx>,
                    Tn::T: Differentiable),
    const_params: (DIM: usize),
    in_type: Tn,
    out_type: Dest,
    numeric_type: Tn::T,
    forward: |input: &Tn, _| input.view().reduce_dim::<Dest, DIM>(|x, y| x + y),
    backward: |in_grad: Tn, _| in_grad.map(|_, x| *x + Tn::T::one()),
});

binary_op!(AddOp<Tn: DifferentiableTensor> {
    args: (),
    where_clauses: ( Tn: for<'a> Add<&'a Tn, Output = Tn> ),
    const_params: (),
    in_type_1: Tn,
    in_type_2: Tn,
    out_type: Tn,
    numeric_type: Tn::T,
    forward: |in1: &Tn, in2: &Tn, _| in1.clone() + in2,
    backward_1: |in_grad: Tn, args: BinaryBackwardArgs<Tn, Tn, Tn, _>| in_grad.map(|idx, in_grad| *in_grad + args.out_grad[idx]),
    backward_2: |in_grad: Tn, args: BinaryBackwardArgs<Tn, Tn, Tn, _>| in_grad.map(|idx, in_grad| *in_grad + args.out_grad[idx]),
});

binary_op!(ElemAddOp<Lhs: DifferentiableTensor, Rhs: DifferentiableTensor> {
    args: (),
    where_clauses: (Rhs: DifferentiableTensor<T = Lhs::T> + Broadcastable<Lhs>,
                    Lhs::T: Differentiable),
    const_params: (),
    in_type_1: Lhs,
    in_type_2: Rhs,
    out_type: Lhs,
    numeric_type: Lhs::T,
    forward: |lhs: &Lhs, rhs: &Rhs, _| {
        lhs.view() + rhs.view().broadcast()
    },
    backward_1: |in_grad: Lhs, args: BinaryBackwardArgs<Lhs, Rhs, Lhs, _>| {
        in_grad.map(|idx, in_grad| *in_grad + args.out_grad[idx])
    },
    backward_2: |in_grad: Rhs, args: BinaryBackwardArgs<Rhs, Lhs, Lhs, _>| {
        let mut in_grad_updated = in_grad;
        for (idx, out_grad) in args.out_grad.iter() {
            let in_idx = Rhs::unbroadcasted_idx(&idx);
            in_grad_updated = in_grad_updated.set(&in_idx, |val| *val + *out_grad);
        }
        in_grad_updated
    },
});

binary_op!(ElemMulOp<Lhs: DifferentiableTensor, Rhs: DifferentiableTensor> {
    args: (),
    where_clauses: (Rhs: DifferentiableTensor<T = Lhs::T> + Broadcastable<Lhs>),
    const_params: (),
    in_type_1: Lhs,
    in_type_2: Rhs,
    out_type: Lhs,
    numeric_type: Lhs::T,
    forward: |lhs: &Lhs, rhs: &Rhs, _| {
        let rhs_bcast = rhs.view().broadcast();
        Lhs::from_fn(|idx| lhs[idx] * rhs_bcast[idx])
    },
    backward_1: (|in_grad: Lhs, args: BinaryBackwardArgs<Lhs, Rhs, Lhs, _>| {
        let rhs = args.other_in_data;
        let rhs_bcast = rhs.view().broadcast();
        in_grad.map(|idx, in_grad| *in_grad + rhs_bcast[idx] * args.out_grad[idx])
    }),
    backward_2: (|mut in_grad: Rhs, args: BinaryBackwardArgs<Rhs, Lhs, Lhs, _>| {
        for (idx, out_grad) in args.out_grad.iter() {
            let in_idx = Rhs::unbroadcasted_idx(&idx);
            in_grad = in_grad.set(&in_idx, |val| *val + args.other_in_data[&idx] * *out_grad);
        }
        in_grad
    }),
});

binary_op!(ScalarMulOp<Tn: DifferentiableTensor> {
    args: (),
    where_clauses: (),
    const_params: (),
    in_type_1: Tn,
    in_type_2: Scalar<Tn::T>,
    out_type: Tn,
    numeric_type: Tn::T,
    forward: |in1: &Tn, in2: &Scalar<Tn::T>, _| in1.clone() * in2.val(),
    backward_1: (|in_grad: Tn, args: BinaryBackwardArgs<Tn, Scalar<Tn::T>, Tn, _>|
                 in_grad.map(|idx, &in_grad| in_grad + args.other_in_data.val() * args.out_grad[idx])),
    backward_2: (|mut in_grad: Scalar<Tn::T>, args: BinaryBackwardArgs<Scalar<Tn::T>, Tn, Tn, _>| {
        for (idx, &other_in_data) in args.other_in_data.iter() {
            in_grad = in_grad.map(|_, &in_grad| in_grad + other_in_data * args.out_grad[&idx]);
        }
        in_grad
    }),
});

binary_op!(MatMulOp<T: Differentiable> {
    args: (),
    where_clauses: (),
    const_params: (M: usize, N: usize, P: usize),
    in_type_1: Matrix<T, M, N>,
    in_type_2: Matrix<T, N, P>,
    out_type: Matrix<T, M, P>,
    numeric_type: T,
    forward: |a: &Matrix<T, M, N>, b: &Matrix<T, N, P>, _| a * b,
    backward_1: |a_grad: Matrix<T, M, N>, args: BinaryBackwardArgs<Matrix<T, M, N>, Matrix<T, N, P>, Matrix<T, M, P>, _>| {
        let b = args.other_in_data;
        a_grad.add_matmul(args.out_grad.view(), b.view().transpose())
    },
    backward_2: |b_grad: Matrix<T, N, P>, args: BinaryBackwardArgs<Matrix<T, N, P>, Matrix<T, M, N>, Matrix<T, M, P>, _>| {
        let a = args.other_in_data;
        b_grad.add_matmul(a.view().transpose(), args.out_grad.view())
    },
});

binary_op!(MatVecMulOp<T: Differentiable> {
    args: (),
    where_clauses: (),
    const_params: (M: usize, N: usize),
    in_type_1: Matrix<T, M, N>,
    in_type_2: Vector<T, N>,
    out_type: Vector<T, M>,
    numeric_type: T,
    forward: |a: &Matrix<T, M, N>, x: &Vector<T, N>, _| a * x,
    backward_1: |a_grad: Matrix<T, M, N>, args: BinaryBackwardArgs<Matrix<T, M, N>, Vector<T, N>, Vector<T, M>, _>| {
        let x = args.other_in_data;
        a_grad.add_matmul(args.out_grad.as_col_vector(), x.as_row_vector())
    },
    backward_2: |x_grad: Vector<T, N>, args: BinaryBackwardArgs<Vector<T, N>, Matrix<T, M, N>, Vector<T, M>, _>| {
        let a = args.other_in_data;
        x_grad.add_matvecmul(a.view().transpose(), args.out_grad.view())
    },
});
