use crate::{
    matrix::{matrix_shape, Matrix},
    numeric::Numeric,
    scalar::Scalar,
    tensor::{num_elems, BasicTensor, Tensor},
};
use num::{traits::real::Real, One, Zero};
use std::fmt::Debug;
use std::marker::PhantomData;

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
        let out = Tn::from_basic(input).relu();
        Box::new(out)
    }
    fn backward<'a>(&self, args: BackwardArgs<Tn::T>) -> BackwardOutput<Tn::T> {
        if let BackwardArgs::Unary {
            in_grad: in_grad_basic,
            out_grad,
            out_data,
            ..
        } = args
        {
            let in_grad: Box<Tn> = Tn::from_basic_boxed(in_grad_basic);
            let updated_grad = in_grad.map(|idx, in_grad| {
                let diff = if out_data[idx.as_ref()] > Tn::T::zero() {
                    out_grad[idx.as_ref()]
                } else {
                    Tn::T::zero()
                };

                in_grad + diff
            });
            BackwardOutput::Unary(Box::new(updated_grad))
        } else {
            unreachable!()
        }
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
        let out = Tn::from_basic(a) + Tn::ref_from_basic(b);
        Box::new(out)
    }
    fn backward<'a>(&self, args: BackwardArgs<Tn::T>) -> BackwardOutput<Tn::T> {
        if let BackwardArgs::Binary {
            in_grad: (in_grad_basic_1, in_grad_basic_2),
            out_grad,
            ..
        } = args
        {
            let in_grad_1: Box<Tn> = Tn::from_basic_boxed(in_grad_basic_1);
            let in_grad_2: Box<Tn> = Tn::from_basic_boxed(in_grad_basic_2);

            let in_grad_1_updated = in_grad_1.map(|idx, in_grad| in_grad + out_grad[idx.as_ref()]);
            let in_grad_2_updated = in_grad_2.map(|idx, in_grad| in_grad + out_grad[idx.as_ref()]);

            BackwardOutput::Binary(Box::new(in_grad_1_updated), Box::new(in_grad_2_updated))
        } else {
            unreachable!()
        }
    }
}

#[derive(Debug)]
pub struct ElemPowOp<Tn: Tensor> {
    _markers: PhantomData<Tn>,
    n: Tn::T,
}

impl<Tn: Tensor> ElemPowOp<Tn> {
    pub fn new(n: Tn::T) -> Box<Self> {
        Box::new(Self {
            _markers: PhantomData,
            n,
        })
    }
}

impl<Tn: Tensor> Op<Tn::T> for ElemPowOp<Tn> {
    fn forward(&self, inputs: ForwardInput<Tn::T>) -> Box<dyn BasicTensor<Tn::T>> {
        let input = inputs.unary();
        let out = Tn::from_fn(|idx| input[idx.as_ref()].powf(self.n));
        Box::new(out)
    }
    fn backward<'a>(&self, args: BackwardArgs<Tn::T>) -> BackwardOutput<Tn::T> {
        if let BackwardArgs::Unary {
            in_grad: in_grad_basic,
            in_data: in_datas,
            out_grad: out_grads,
            ..
        } = args
        {
            let in_grad: Box<Tn> = Tn::from_basic_boxed(in_grad_basic);
            let updated_grad = in_grad.map(|idx, in_grad| {
                let out_grad = out_grads[idx.as_ref()];
                let in_data = in_datas[idx.as_ref()];

                in_grad + ((self.n * in_data.powf(self.n - Tn::T::one())) * out_grad)
            });
            BackwardOutput::Unary(Box::new(updated_grad))
        } else {
            unreachable!()
        }
    }
}

#[derive(Debug)]
pub struct SumOp<Tn: Tensor> {
    _markers: PhantomData<Tn>,
}

impl<Tn: Tensor> SumOp<Tn> {
    pub fn new() -> Box<Self> {
        Box::new(Self {
            _markers: PhantomData,
        })
    }
}

impl<Tn: Tensor> Op<Tn::T> for SumOp<Tn> {
    fn forward(&self, inputs: ForwardInput<Tn::T>) -> Box<dyn BasicTensor<Tn::T>> {
        let input_basic = inputs.unary();
        let input = Tn::ref_from_basic(input_basic);
        let out = input.sum();

        Box::new(out)
    }
    fn backward<'a>(&self, args: BackwardArgs<Tn::T>) -> BackwardOutput<Tn::T> {
        if let BackwardArgs::Unary {
            in_grad: in_grad_basic,
            ..
        } = args
        {
            let in_grad = *Tn::from_basic_boxed(in_grad_basic);

            BackwardOutput::Unary(Box::new(in_grad.map(|_, v| v + Tn::T::one())))
        } else {
            unreachable!()
        }
    }
}

#[derive(Debug)]
pub struct ElemMulOp<Tn: Tensor> {
    _markers: PhantomData<Tn>,
}

impl<Tn: Tensor> ElemMulOp<Tn> {
    pub fn new() -> Box<Self> {
        Box::new(Self {
            _markers: PhantomData,
        })
    }
}

impl<Tn: Tensor> Op<Tn::T> for ElemMulOp<Tn> {
    fn forward(&self, inputs: ForwardInput<Tn::T>) -> Box<dyn BasicTensor<Tn::T>> {
        let (a, b) = inputs.binary();
        let out = Tn::from_fn(|idx| a[idx.as_ref()] * b[idx.as_ref()]);

        Box::new(out)
    }
    fn backward<'a>(&self, args: BackwardArgs<Tn::T>) -> BackwardOutput<Tn::T> {
        if let BackwardArgs::Binary {
            in_grad: (in_grad_basic_1, in_grad_basic_2),
            in_data: (in_data_1, in_data_2),
            out_grad,
            ..
        } = args
        {
            let in_grad_1: Box<Tn> = Tn::from_basic_boxed(in_grad_basic_1);
            let in_grad_2: Box<Tn> = Tn::from_basic_boxed(in_grad_basic_2);

            let in_grad_1_updated = in_grad_1
                .map(|idx, in_grad| in_grad + in_data_2[idx.as_ref()] * out_grad[idx.as_ref()]);
            let in_grad_2_updated = in_grad_2
                .map(|idx, in_grad| in_grad + in_data_1[idx.as_ref()] * out_grad[idx.as_ref()]);

            BackwardOutput::Binary(Box::new(in_grad_1_updated), Box::new(in_grad_2_updated))
        } else {
            unreachable!()
        }
    }
}

#[derive(Debug)]
pub struct ScalarMulOp<Tn: Tensor> {
    _markers: (PhantomData<Tn>, PhantomData<Scalar<Tn::T>>),
}

impl<Tn: Tensor> ScalarMulOp<Tn> {
    pub fn new() -> Box<Self> {
        Box::new(Self {
            _markers: (PhantomData, PhantomData),
        })
    }
}

impl<Tn: Tensor> Op<Tn::T> for ScalarMulOp<Tn> {
    fn forward(&self, inputs: ForwardInput<Tn::T>) -> Box<dyn BasicTensor<Tn::T>> {
        let (a_basic, b_basic) = inputs.binary();
        let a: Tn = Tn::from_basic(a_basic);
        let b: &Scalar<Tn::T> = Scalar::ref_from_basic(b_basic);

        Box::new(a * b.val())
    }
    fn backward<'a>(&self, args: BackwardArgs<Tn::T>) -> BackwardOutput<Tn::T> {
        if let BackwardArgs::Binary {
            in_grad: (in_grad_basic_1, in_grad_basic_2),
            in_data: (in_data_1, in_data_2_basic),
            out_grad,
            ..
        } = args
        {
            let in_grad_1: Box<Tn> = Tn::from_basic_boxed(in_grad_basic_1);
            let in_grad_2: Scalar<Tn::T> = *Scalar::from_basic_boxed(in_grad_basic_2);
            let in_data_2: &Scalar<Tn::T> = Scalar::ref_from_basic(in_data_2_basic);

            let mut in_grad_2_updated = in_grad_2;
            for (idx, _) in in_grad_1.iter() {
                in_grad_2_updated = in_grad_2_updated
                    .map(|_, in_grad| in_grad + in_data_1[idx.as_ref()] * out_grad[idx.as_ref()]);
            }
            let in_grad_1_updated =
                in_grad_1.map(|idx, in_grad| in_grad + in_data_2.val() * out_grad[idx.as_ref()]);

            BackwardOutput::Binary(Box::new(in_grad_1_updated), Box::new(in_grad_2_updated))
        } else {
            unreachable!()
        }
    }
}

#[derive(Debug)]
pub struct MatMulOp<T: Numeric, const M: usize, const N: usize, const P: usize> {
    _markers: (PhantomData<T>,),
}

impl<T: Numeric, const M: usize, const N: usize, const P: usize> MatMulOp<T, M, N, P> {
    pub fn new() -> Box<Self> {
        Box::new(Self {
            _markers: (PhantomData,),
        })
    }
}

impl<T: Numeric, const M: usize, const N: usize, const P: usize> Op<T> for MatMulOp<T, M, N, P>
where
    [(); num_elems(2, matrix_shape(M, N))]:,
    [(); num_elems(2, matrix_shape(N, M))]:,
    [(); num_elems(2, matrix_shape(N, P))]:,
    [(); num_elems(2, matrix_shape(P, N))]:,
    [(); num_elems(2, matrix_shape(M, P))]:,
{
    fn forward(&self, input: ForwardInput<T>) -> Box<dyn BasicTensor<T>> {
        let (a_basic, b_basic) = input.binary();
        let a = Matrix::<T, M, N>::ref_from_basic(a_basic);
        let b = Matrix::<T, N, P>::ref_from_basic(b_basic);
        let out = a * b;
        Box::new(out)
    }
    fn backward(&self, args: BackwardArgs<T>) -> BackwardOutput<T> {
        if let BackwardArgs::Binary {
            in_grad: (a_grad_basic, b_grad_basic),
            in_data: (a_basic, b_basic),
            out_grad: out_grad_basic,
            ..
        } = args
        {
            let a = Matrix::<T, M, N>::ref_from_basic(a_basic);
            let a_grad = *Matrix::<T, M, N>::from_basic_boxed(a_grad_basic);

            let b = Matrix::<T, N, P>::ref_from_basic(b_basic);
            let b_grad = *Matrix::<T, N, P>::from_basic_boxed(b_grad_basic);

            let out_grad = Matrix::<T, M, P>::ref_from_basic(out_grad_basic);

            let a_diff = out_grad * b.view().transpose();
            let b_diff = a.view().transpose() * out_grad;

            let a_grad_updated = a_grad + &a_diff;
            let b_grad_updated = b_grad + &b_diff;

            BackwardOutput::Binary(Box::new(a_grad_updated), Box::new(b_grad_updated))
        } else {
            unreachable!()
        }
    }
}
