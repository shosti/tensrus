use crate::{
    broadcast::{broadcast_compat, BroadcastableTo},
    generic_tensor::GenericTensor,
    matrix::Matrix,
    numeric::Numeric,
    scalar::Scalar,
    shape::{reduced_shape, Shaped},
    storage::TensorStorage,
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
pub struct DimSumOp<Src: Tensor, Dest: Tensor<T = Src::T>, const DIM: usize> {
    _markers: PhantomData<(Src, Dest)>,
}

impl<Src: Tensor, Dest: Tensor<T = Src::T>, const DIM: usize> DimSumOp<Src, Dest, DIM> {
    pub fn new() -> Box<Self> {
        Box::new(Self {
            _markers: PhantomData,
        })
    }
}

impl<Src: Tensor, Dest: Tensor, const DIM: usize> Op<Src::T> for DimSumOp<Src, Dest, DIM>
where
    Src: Shaped + TensorStorage<Src::T>,
    Src::Idx: From<[usize; Src::R]>,
    Dest: Tensor<T = Src::T>
        + Shaped<R = { Src::R }, S = { reduced_shape(Src::R, Src::S, DIM) }>
        + From<GenericTensor<Src::T, { Src::R }, { reduced_shape(Src::R, Src::S, DIM) }>>,
{
    fn forward(&self, args: ForwardInput<Src::T>) -> Box<dyn BasicTensor<Src::T>> {
        let input = args.unary();
        let in_typed = Src::ref_from_basic(input);
        let reduced: Dest = in_typed.view().reduce_dim::<DIM>(|x, y| x + y).into();
        Box::new(reduced)
    }

    fn backward(&self, args: BackwardArgs<Src::T>) -> BackwardOutput<Src::T> {
        if let BackwardArgs::Unary {
            in_grad: in_grad_basic,
            ..
        } = args
        {
            let in_grad = *Src::from_basic_boxed(in_grad_basic);
            let in_grad_updated = in_grad.map(|_, x| x + Src::T::one());

            BackwardOutput::Unary(Box::new(in_grad_updated))
        } else {
            unreachable!()
        }
    }
}

#[derive(Debug)]
pub struct BCastMulOp<Tn, Rhs> {
    _markers: PhantomData<(Tn, Rhs)>,
}

impl<Tn, Rhs> BCastMulOp<Tn, Rhs> {
    pub fn new() -> Box<Self> {
        Box::new(Self {
            _markers: PhantomData,
        })
    }
}

impl<Tn, Rhs> Op<Tn::T> for BCastMulOp<Tn, Rhs>
where
    Tn: Tensor + Shaped,
    Rhs: Tensor<T = Tn::T> + Shaped,
    Assert<{ broadcast_compat(Rhs::R, Rhs::S, Tn::R, Tn::S) }>: IsTrue,
{
    fn forward(&self, _args: ForwardInput<Tn::T>) -> Box<dyn BasicTensor<Tn::T>> {
        // let (a_untyped, b_untyped) = args.binary();
        // let a = Tn::ref_from_basic(a_untyped);
        // let b_orig = Rhs::ref_from_basic(b_untyped);
        // let b = b_orig.broadcast::<{ Tn::R }, { Tn::S }>();
        todo!()
        // let out = a.map(|idx, x| x * b[&idx]);
        // Box::new(out)
    }

    fn backward(&self, _args: BackwardArgs<Tn::T>) -> BackwardOutput<Tn::T> {
        todo!()
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

binary_op!(ElemMulOp<Tn: Tensor, Rhs: Tensor> {
    args: (),
    where_clauses: (Rhs: Tensor<T = Tn::T> + for<'a> BroadcastableTo<'a, Tn::T, Tn>,
                    Tn: Shaped,
                    Assert<{ broadcast_compat(Rhs::R, Rhs::S, Tn::R, Tn::S) }>: IsTrue),
    const_params: (),
    in_type_1: Tn,
    in_type_2: Rhs,
    out_type: Tn,
    numeric_type: Tn::T,
    forward: |in1: &Tn, in2: &Rhs, _| {
        let other: View<Tn> = in2.broadcast();
        Tn::from_fn(|idx| in1[idx] * other[idx])
    },
    backward_1: (|in_grad: Tn, args: BinaryBackwardArgs<Tn, Rhs, Tn, _>|
                 todo!()),
                 // in_grad.map(|idx, in_grad| in_grad + args.other_in_data[idx] * args.out_grad[idx])),
    backward_2: (|in_grad: Rhs, args: BinaryBackwardArgs<Rhs, Tn, Tn, _>|
                 todo!()),
                 // in_grad.map(|idx, in_grad| in_grad + args.other_in_data[idx] * args.out_grad[idx])),
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

binary_op!(MatMulOp<T: Numeric> {
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
        args.out_grad.matmul_view_into(b.matrix_view().transpose(), a_grad)
    },
    backward_2: |b_grad: Matrix<T, N, P>, args: BinaryBackwardArgs<Matrix<T, N, P>, Matrix<T, M, N>, Matrix<T, M, P>, _>| {
        let a = args.other_in_data;
        a.matrix_view().transpose().matmul_into(args.out_grad, b_grad)
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
        a.matrix_view().transpose().matvecmul_into(args.out_grad, x_grad)
    },
});
