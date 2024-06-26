use crate::{
    broadcast::broadcast_compat,
    generic_tensor::{AsGeneric, GenericTensor},
    matrix::{Matrix, MatrixLike},
    numeric::Numeric,
    scalar::Scalar,
    shape::{reduced_shape, Shape},
    storage::num_elems,
    tensor::{BasicTensor, Tensor},
    type_assert::{Assert, IsTrue},
    vector::Vector,
    view::View,
};
use num::{traits::real::Real, One, Zero};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Add;

#[derive(Debug)]
pub enum ForwardInput<'a, T: Numeric> {
    Unary(&'a dyn BasicTensor<T>),
    Binary(&'a dyn BasicTensor<T>, &'a dyn BasicTensor<T>),
}

impl<'a, T: Numeric> ForwardInput<'a, T> {
    pub fn unary(&'a self) -> &'a dyn BasicTensor<T> {
        if let Self::Unary(t) = self {
            *t
        } else {
            unreachable!()
        }
    }

    pub fn binary(&'a self) -> (&'a dyn BasicTensor<T>, &'a dyn BasicTensor<T>) {
        if let Self::Binary(t1, t2) = self {
            (*t1, *t2)
        } else {
            unreachable!()
        }
    }
}

#[derive(Debug)]
pub enum BackwardArgs<'a, T: Numeric> {
    Unary {
        in_grad: Box<dyn BasicTensor<T>>,
        in_data: &'a dyn BasicTensor<T>,
        out_grad: &'a dyn BasicTensor<T>,
        out_data: &'a dyn BasicTensor<T>,
    },
    Binary {
        in_grad: (Box<dyn BasicTensor<T>>, Box<dyn BasicTensor<T>>),
        in_data: (&'a dyn BasicTensor<T>, &'a dyn BasicTensor<T>),
        out_grad: &'a dyn BasicTensor<T>,
        out_data: &'a dyn BasicTensor<T>,
    },
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
            unreachable!()
        }
    }

    pub fn binary(self) -> (Box<dyn BasicTensor<T>>, Box<dyn BasicTensor<T>>) {
        if let Self::Binary(out1, out2) = self {
            (out1, out2)
        } else {
            unreachable!()
        }
    }
}

pub trait Op<T: Numeric>: Debug {
    fn forward(&self, input: ForwardInput<T>) -> Box<dyn BasicTensor<T>>;
    fn backward(&self, args: BackwardArgs<T>) -> BackwardOutput<T>;
}

#[derive(Debug)]
pub struct DimSumOp<T: Numeric, const R: usize, const S: Shape, const DIM: usize> {
    _markers: PhantomData<GenericTensor<T, R, S>>,
}

impl<T: Numeric, const R: usize, const S: Shape, const DIM: usize> DimSumOp<T, R, S, DIM> {
    pub fn new() -> Box<Self> {
        Box::new(Self {
            _markers: PhantomData,
        })
    }
}

impl<T: Numeric, const R: usize, const S: Shape, const DIM: usize> Op<T> for DimSumOp<T, R, S, DIM>
where
    [(); num_elems(R, reduced_shape(R, S, DIM))]:,
{
    fn forward(&self, args: ForwardInput<T>) -> Box<dyn BasicTensor<T>> {
        let input = args.unary();
        let in_typed = GenericTensor::<T, R, S>::ref_from_basic(input);
        let reduced: GenericTensor<T, R, { reduced_shape(R, S, DIM) }> =
            in_typed.view().reduce_dim::<DIM>(|x, y| x + y);
        Box::new(reduced)
    }

    fn backward(&self, args: BackwardArgs<T>) -> BackwardOutput<T> {
        if let BackwardArgs::Unary {
            in_grad: in_grad_basic,
            ..
        } = args
        {
            let in_grad = *GenericTensor::<T, R, S>::from_basic_boxed(in_grad_basic);
            let in_grad_updated = in_grad.map(|_, x| x + T::one());

            BackwardOutput::Unary(Box::new(in_grad_updated))
        } else {
            unreachable!()
        }
    }
}

macro_rules! unary_op {
    ($name:ident < $( $generic:ident : $subtype:ident ),* > {
        args: ( $( $arg:ident : $argty:ty ),* ),
        in_type: $inty:ty,
        out_type: $outty:ty,
        numeric_type: $numty:ty,
        forward: $forward:expr,
        backward: $backward:expr,
    }) => {

        #[derive(Debug)]
        pub struct $name< $( $generic : $subtype ),* > {
            _markers: ( $( std::marker::PhantomData< $generic > ,)* ),
            args: ( $( $argty ,)* ),
        }

        impl< $( $generic : $subtype ),* > $name< $( $generic ),* > {
            pub fn new( $( $arg : $argty ),* ) -> Box<Self> {
                Box::new(Self {
                    _markers: ( $(PhantomData::< $generic >,)* ),
                    args: ( $( $arg ,)* ),
                })
            }
        }

        impl< $( $generic : $subtype ),* > Op< $numty > for $name< $( $generic ),* > {
            fn forward(&self, inputs: ForwardInput< $numty >) -> Box<dyn BasicTensor< $numty >> {
                let input = <$inty>::ref_from_basic(inputs.unary());
                let out: $outty = ($forward)(input, self.args);
                Box::new(out)
            }
            fn backward<'a>(&self, args: BackwardArgs< $numty >) -> BackwardOutput< $numty > {
                if let BackwardArgs::Unary {
                    in_grad: in_grad_basic,
                    in_data: in_data_basic,
                    out_grad: out_grad_basic,
                    out_data: out_data_basic,
                } = args
                {
                    let in_grad = *<$inty>::from_basic_boxed(in_grad_basic);
                    let in_data = <$inty>::ref_from_basic(in_data_basic);
                    let out_grad = <$outty>::ref_from_basic(out_grad_basic);
                    let out_data = <$outty>::ref_from_basic(out_data_basic);
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
        pub struct $name< $( $generic : $subtype ),* $(, const $constparam : $constparamty )* > {
            _markers: ( $( std::marker::PhantomData< $generic > ,)* ),
            args: ( $( $argty ,)* ),
        }

        impl< $( $generic : $subtype ),* $(, const $constparam : $constparamty )* > $name< $( $generic ),* $(, $constparam )* > {
            pub fn new( $( $arg : $argty ),* ) -> Box<Self> {
                Box::new(Self {
                    _markers: ( $(PhantomData::< $generic >,)* ),
                    args: ( $( $arg ,)* ),
                })
            }
        }

        impl< $( $generic : $subtype ),* $(, const $constparam : $constparamty )* > Op< $numty > for $name< $( $generic ),* $(, $constparam )* >
            where [(); 0]:,
            $( $whereclauses )*
        {
            fn forward(&self, inputs: ForwardInput< $numty >) -> Box<dyn BasicTensor< $numty >> {
                let (input_1_basic, input_2_basic) = inputs.binary();
                let input_1 = <$inty1>::ref_from_basic(input_1_basic);
                let input_2 = <$inty2>::ref_from_basic(input_2_basic);
                let out: $outty = ($forward)(input_1, input_2, self.args);

                Box::new(out)
            }

            fn backward<'a>(&self, args: BackwardArgs< $numty >) -> BackwardOutput< $numty > {
                if let BackwardArgs::Binary {
                    in_grad: (in_grad_basic_1, in_grad_basic_2),
                    in_data: (in_data_basic_1, in_data_basic_2),
                    out_grad: out_grad_basic,
                    out_data: out_data_basic,
                } = args
                {
                    let in_grad_1 = *<$inty1>::from_basic_boxed(in_grad_basic_1);
                    let in_grad_2 = *<$inty2>::from_basic_boxed(in_grad_basic_2);
                    let in_data_1 = <$inty1>::ref_from_basic(in_data_basic_1);
                    let in_data_2 = <$inty2>::ref_from_basic(in_data_basic_2);
                    let out_grad = <$outty>::ref_from_basic(out_grad_basic);
                    let out_data = <$outty>::ref_from_basic(out_data_basic);

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

unary_op!(ReLUOp<Tn: Tensor> {
    args: (),
    in_type: Tn,
    out_type: Tn,
    numeric_type: Tn::T,
    forward: |input: &Tn, _args: ()| input.clone().relu(),
    backward: (|in_grad: Tn, args: UnaryBackwardArgs<Tn, Tn, _>|
               in_grad.map(|idx, in_grad| {
                   let diff = if args.out_data[idx] > Tn::T::zero() {
                       args.out_grad[idx.as_ref()]
                   } else {
                       Tn::T::zero()
                   };

                   in_grad + diff
               })),
});

unary_op!(ElemLnOp<Tn: Tensor> {
    args: (),
    in_type: Tn,
    out_type: Tn,
    numeric_type: Tn::T,
    forward: |input: &Tn, _args: ()| input.clone().map(|_, x| x.ln()),
    backward: (|in_grad: Tn, args: UnaryBackwardArgs<Tn, Tn, _>|
               in_grad.map(|idx, in_grad| {
                   in_grad + (Tn::T::one() / args.in_data[idx])
               })),
});

unary_op!(ElemPowOp<Tn: Tensor> {
    args: (n: Tn::T),
    in_type: Tn,
    out_type: Tn,
    numeric_type: Tn::T,
    forward: |input: &Tn, (n,): (Tn::T,)| Tn::from_fn(|idx| input[idx.as_ref()].powf(n)),
    backward: (|in_grad: Tn, args: UnaryBackwardArgs<Tn, Tn, (Tn::T,)>|
               in_grad.map(|idx, in_grad| {
                   let out_grad_val = args.out_grad[idx];
                   let in_data_val = args.in_data[idx];
                   let n = args.args.0;

                   in_grad + ((n * in_data_val.powf(n - Tn::T::one())) * out_grad_val)
               })),
});

unary_op!(SumOp<Tn: Tensor> {
    args: (),
    in_type: Tn,
    out_type: Scalar<Tn::T>,
    numeric_type: Tn::T,
    forward: |input: &Tn, _| input.sum(),
    backward: |in_grad: Tn, _| in_grad.map(|_, v| v + Tn::T::one()),
});

binary_op!(AddOp<Tn: Tensor> {
    args: (),
    where_clauses: ( Tn: for<'a> Add<&'a Tn, Output = Tn> ),
    const_params: (),
    in_type_1: Tn,
    in_type_2: Tn,
    out_type: Tn,
    numeric_type: Tn::T,
    forward: |in1: &Tn, in2: &Tn, _| in1.clone() + in2,
    backward_1: |in_grad: Tn, args: BinaryBackwardArgs<Tn, Tn, Tn, _>| in_grad.map(|idx, in_grad| in_grad + args.out_grad[idx]),
    backward_2: |in_grad: Tn, args: BinaryBackwardArgs<Tn, Tn, Tn, _>| in_grad.map(|idx, in_grad| in_grad + args.out_grad[idx]),
});

binary_op!(ElemAddOp<T: Numeric> {
    args: (),
    where_clauses: (Assert<{ broadcast_compat(R_RHS, S_RHS, R, S) }>: IsTrue),
    const_params: (R: usize, S: Shape, R_RHS: usize, S_RHS: Shape),
    in_type_1: GenericTensor<T, R, S>,
    in_type_2: GenericTensor<T, R_RHS, S_RHS>,
    out_type: GenericTensor<T, R, S>,
    numeric_type: T,
    forward: |in1: &GenericTensor<T, R, S>, in2: &GenericTensor<T, R_RHS, S_RHS>, _| {
        let v = in2.view();
        let other: View<GenericTensor<T, R, S>> = v.broadcast();
        // This _should_ be just in1.clone() + other, but there's compiler bug
        // (I think there are actually some overlapping trait implementations
        // somewhere). See https://github.com/rust-lang/rust/issues/119692
        in1.clone().map(|idx, x| x + other[idx])
    },
    backward_1: |in_grad: GenericTensor<T, R, S>, args: BinaryBackwardArgs<GenericTensor<T, R, S>, GenericTensor<T, R_RHS, S_RHS>, GenericTensor<T, R, S>, _>| {
        in_grad.map(|idx, in_grad| in_grad + args.out_grad[idx])
    },
    backward_2: |in_grad: GenericTensor<T, R_RHS, S_RHS>, args: BinaryBackwardArgs<GenericTensor<T, R_RHS, S_RHS>, GenericTensor<T, R, S>, GenericTensor<T, R, S>, _>| {
        let mut in_grad_updated = in_grad;
        for (idx, out_grad) in args.out_grad.iter() {
            let in_idx = View::<GenericTensor<T, R_RHS, S_RHS>>::unbroadcasted_idx::<R, S>(&idx);
            in_grad_updated = in_grad_updated.set(&in_idx, |val| val + out_grad);
        }
        in_grad_updated
    },
});

binary_op!(ElemMulOp<Lhs: Tensor, Rhs: Tensor> {
    args: (),
    where_clauses: (Assert<{ broadcast_compat(R_RHS, S_RHS, R, S) }>: IsTrue,
                    Lhs: AsGeneric<Lhs::T, R, S>,
                    Rhs: AsGeneric<Lhs::T, R_RHS, S_RHS> + Tensor<T = Lhs::T>),
    const_params: (R: usize, S: Shape, R_RHS: usize, S_RHS: Shape),
    in_type_1: Lhs,
    in_type_2: Rhs,
    out_type: GenericTensor<Lhs::T, R, S>,
    numeric_type: Lhs::T,
    forward: |in1: &Lhs, in2: &Rhs, _| {
        let lhs = in1.as_generic();
        let rhs_orig = in2.as_generic();
        let rhs: View<GenericTensor<_, R, S>> = rhs_orig.broadcast();
        GenericTensor::from_fn(|idx| lhs[idx] * rhs[idx])
    },
    backward_1: (|in_grad: Lhs, args: BinaryBackwardArgs<Lhs, Rhs, GenericTensor<Lhs::T, R, S>, _>| {
        let rhs_orig = args.other_in_data.as_generic();
        let rhs: View<GenericTensor<_, R, S>> = rhs_orig.broadcast();
        let mut in_grad_gen: GenericTensor<_, R, S> = in_grad.into();
        in_grad_gen = in_grad_gen.map(|idx, in_grad| in_grad + rhs[idx] * args.out_grad[idx]);
        in_grad_gen.into()
    }),
    backward_2: (|in_grad: Rhs, args: BinaryBackwardArgs<Rhs, Lhs, GenericTensor<Lhs::T, R, S>, _>| {
        let mut in_grad_updated: GenericTensor<Lhs::T, R_RHS, S_RHS> = in_grad.into();
        let out_grad_gen = args.out_grad.as_generic();
        let lhs_in_data = args.other_in_data.as_generic();
        for (idx, out_grad) in out_grad_gen.iter() {
            let in_idx = View::<GenericTensor<Lhs::T, R_RHS, S_RHS>>::unbroadcasted_idx::<R, S>(&idx);
            in_grad_updated = in_grad_updated.set(&in_idx, |val| val + lhs_in_data[&idx] * out_grad)
        }
        in_grad_updated.into()
    }),
});

binary_op!(ScalarMulOp<Tn: Tensor> {
    args: (),
    where_clauses: (),
    const_params: (),
    in_type_1: Tn,
    in_type_2: Scalar<Tn::T>,
    out_type: Tn,
    numeric_type: Tn::T,
    forward: |in1: &Tn, in2: &Scalar<Tn::T>, _| in1.clone() * in2.val(),
    backward_1: (|in_grad: Tn, args: BinaryBackwardArgs<Tn, Scalar<Tn::T>, Tn, _>|
                 in_grad.map(|idx, in_grad| in_grad + args.other_in_data.val() * args.out_grad[idx])),
    backward_2: (|mut in_grad: Scalar<Tn::T>, args: BinaryBackwardArgs<Scalar<Tn::T>, Tn, Tn, _>| {
        for (idx, other_in_data) in args.other_in_data.iter() {
            in_grad = in_grad.map(|_, in_grad| in_grad + other_in_data * args.out_grad[idx.as_ref()]);
        }
        in_grad
    }),
});

binary_op!(MatMulOp<Lhs: Tensor, Rhs: Tensor> {
    args: (),
    where_clauses: (Lhs: MatrixLike<Lhs::T, M, N> + From<Matrix<Lhs::T, M, N>>, Rhs: Tensor<T = Lhs::T> + MatrixLike<Lhs::T, N, P> + From<Matrix<Lhs::T, N, P>>),
    const_params: (M: usize, N: usize, P: usize),
    in_type_1: Lhs,
    in_type_2: Rhs,
    out_type: Matrix<Lhs::T, M, P>,
    numeric_type: Lhs::T,
    forward: |a: &Lhs, b: &Rhs, _| a.as_matrix() * b.as_matrix(),
    backward_1: |a_grad: Lhs, args: BinaryBackwardArgs<Lhs, Rhs, Matrix<Lhs::T, M, P>, _>| {
        let b = args.other_in_data;
        let a_grad_mat = args.out_grad.matmul_view_into(b.as_matrix().transpose(), a_grad.into_matrix());
        a_grad_mat.into()
    },
    backward_2: |b_grad: Rhs, args: BinaryBackwardArgs<Rhs, Lhs, Matrix<Lhs::T, M, P>, _>| {
        let a = args.other_in_data;
        let b_grad_mat = a.as_matrix().transpose().matmul_into(args.out_grad, b_grad.into_matrix());
        b_grad_mat.into()
    },
});

binary_op!(MatVecMulOp<T: Numeric> {
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
        args.out_grad.as_col_vector().matmul_view_into(x.as_row_vector(), a_grad)
    },
    backward_2: |x_grad: Vector<T, N>, args: BinaryBackwardArgs<Vector<T, N>, Matrix<T, M, N>, Vector<T, M>, _>| {
        let a = args.other_in_data;
        a.as_matrix().transpose().matvecmul_into(args.out_grad, x_grad)
    },
});
