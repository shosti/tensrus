use num::One;

use self::output::Output;
use self::var_ref::VarRef;
use self::{input::Input, param::Param};
use crate::differentiable::{Differentiable, DifferentiableTensor};
use crate::errors::GraphError;
use crate::matrix::Matrix;
use crate::op::{
    AddOp, DimSumOp, ElemAddOp, ElemLnOp, ElemMulOp, ElemPowOp, MatMulOp, MatVecMulOp, Op, ReLUOp,
    ScalarMulOp, SumOp,
};
use crate::scalar::Scalar;
use crate::shape::{Broadcastable, Reducible};
use crate::vector::Vector;
use std::cell::Ref;
use std::cell::RefCell;

mod graph;
pub mod input;
pub mod output;
pub mod param;
mod var_ref;

pub type Id = u64;

thread_local!(static NEXT_ID: RefCell<Id> = const { RefCell::new(1) });

#[derive(Debug, Clone)]
pub enum Var<Tn> {
    Param(Param<Tn>),
    Input(Input<Tn>),
    Output(Output<Tn>),
}

#[derive(Debug, Copy, Clone)]
enum Children {
    Unary(Id),
    Binary(Id, Id),
}

impl<Tn> Var<Tn>
where
    Tn: DifferentiableTensor,
    Tn::T: Differentiable,
{
    pub fn param(data: Tn) -> Self {
        let inner = Param::new(Self::next_id(), data);
        Self::Param(inner)
    }

    pub fn input(data: Tn) -> Self {
        let inner = Input::new(Self::next_id(), data);
        Self::Input(inner)
    }

    pub fn id(&self) -> Id {
        match self {
            Self::Param(inner) => inner.id(),
            Self::Input(inner) => inner.id(),
            Self::Output(inner) => inner.id(),
        }
    }

    pub fn data(&self) -> Ref<Tn> {
        match self {
            Self::Param(inner) => inner.data(),
            Self::Input(inner) => inner.data(),
            Self::Output(inner) => inner.data(),
        }
    }

    /// Returns true if self is a copy of the same Var (with shared underlying
    /// storage) as other.
    pub fn is(&self, other: &Self) -> bool {
        self.id() == other.id()
    }

    /// Returns self as a Param if self is a parameter, otherwise None.
    pub fn as_param(&self) -> Option<&Param<Tn>> {
        if let Self::Param(p) = self {
            Some(p)
        } else {
            None
        }
    }

    /// Returns the underlying data tensor if there are no existing clones of
    /// the Var. If there existing clones, the existing variable is dropped and
    /// None is returned. See also: Rc.into_inner.
    pub fn into_data(self) -> Option<Tn> {
        match self {
            Self::Param(inner) => inner.into_data(),
            Self::Input(inner) => inner.into_data(),
            Self::Output(inner) => inner.into_data(),
        }
    }

    pub fn sum_elems(&self) -> Var<Scalar<Tn::T>> {
        let op = SumOp::<Tn>::new();

        self.unary_output(op)
    }

    pub fn relu(&self) -> Self {
        let op = ReLUOp::<Tn>::new();

        self.unary_output(op)
    }

    pub fn elem_pow(&self, n: Tn::T) -> Self {
        let op = ElemPowOp::<Tn>::new(n);

        self.unary_output(op)
    }

    pub fn elem_ln(&self) -> Self {
        let op = ElemLnOp::<Tn>::new();

        self.unary_output(op)
    }

    pub fn elem_add<Rhs>(&self, rhs: Var<Rhs>) -> Self
    where
        Rhs: DifferentiableTensor<T = Tn::T> + Broadcastable<Tn>,
        Tn::T: Differentiable,
    {
        let op = ElemAddOp::<Tn, Rhs>::new();

        self.binary_output(&rhs, op)
    }

    pub fn elem_mul<Rhs>(&self, rhs: Var<Rhs>) -> Self
    where
        Rhs: DifferentiableTensor<T = Tn::T> + Broadcastable<Tn>,
        Tn::T: Differentiable,
    {
        let op = ElemMulOp::<Tn, Rhs>::new();

        self.binary_output(&rhs, op)
    }

    pub fn elem_div<Rhs>(&self, rhs: Var<Rhs>) -> Self
    where
        Rhs: DifferentiableTensor<T = Tn::T> + Broadcastable<Tn>,
        Tn::T: Differentiable,
    {
        if self.id() == rhs.id() {
            todo!("haven't implemented divide by self")
        }

        let inv = rhs.elem_pow(-Tn::T::one());
        self.elem_mul(inv)
    }

    pub(self) fn as_ref(&self) -> VarRef {
        match self {
            Self::Param(inner) => VarRef::Param(inner.inner.clone()),
            Self::Input(inner) => VarRef::Input(inner.inner.clone()),
            Self::Output(inner) => VarRef::Output(inner.inner.clone()),
        }
    }

    fn unary_output<Out>(&self, op: Box<dyn Op>) -> Var<Out>
    where
        Out: DifferentiableTensor,
        Out::T: Differentiable,
    {
        let id = Self::next_id();

        Var::Output(Output::new_unary(id, op, self.as_ref()))
    }

    fn binary_output<Rhs, Out>(&self, rhs: &Var<Rhs>, op: Box<dyn Op>) -> Var<Out>
    where
        Rhs: DifferentiableTensor,
        Rhs::T: Differentiable,
        Out: DifferentiableTensor,
        Out::T: Differentiable,
    {
        debug_assert!(self.id() != rhs.id());
        let id = Self::next_id();

        Var::Output(Output::new_binary(id, op, self.as_ref(), rhs.as_ref()))
    }

    fn next_id() -> Id {
        let mut id = 0;
        NEXT_ID.with(|n| {
            id = *n.borrow();
            *n.borrow_mut() = id + 1;
        });

        id
    }

    pub fn dim_sum<const DIM: usize>(&self) -> Var<Tn::Reduced>
    where
        Tn: Reducible<DIM>,
        Tn::Reduced: DifferentiableTensor,
    {
        let op = DimSumOp::<Tn, DIM>::new();

        self.unary_output(op)
    }

    pub fn softmax(&self) -> Self
    where
        Tn: Reducible<1>,
        Tn::Reduced: DifferentiableTensor + Broadcastable<Tn>,
    {
        let row_sums = self.dim_sum::<1>();
        self.elem_div(row_sums)
    }
}

impl<T: Differentiable> Var<Scalar<T>> {
    pub fn backward(&self) -> Result<(), GraphError> {
        match self {
            Self::Param(p) => {
                p.init_grad();
                Ok(())
            }
            Self::Output(o) => o.backward(),
            Self::Input(_) => Ok(()), // backward for an input is a no-op
        }
    }
}

impl<Tn> std::ops::Mul<Var<Scalar<Tn::T>>> for Var<Tn>
where
    Tn: DifferentiableTensor,
    Tn::T: Differentiable,
{
    type Output = Self;

    fn mul(self, rhs: Var<Scalar<Tn::T>>) -> Self {
        if self.id() == rhs.id() {
            return self.elem_pow(Tn::T::two());
        }
        let op = ScalarMulOp::<Tn>::new();

        self.binary_output(&rhs, op)
    }
}

impl<T, const M: usize, const N: usize, const P: usize> std::ops::Mul<Var<Matrix<T, N, P>>>
    for Var<Matrix<T, M, N>>
where
    T: Differentiable,
{
    type Output = Var<Matrix<T, M, P>>;

    fn mul(self, rhs: Var<Matrix<T, N, P>>) -> Var<Matrix<T, M, P>> {
        if self.id() == rhs.id() {
            todo!("matrix squaring not yet implemented")
        }
        let op = MatMulOp::<T, M, N, P>::new();

        self.binary_output(&rhs, op)
    }
}

impl<T, const M: usize, const N: usize> std::ops::Mul<Var<Vector<T, N>>> for Var<Matrix<T, M, N>>
where
    T: Differentiable,
{
    type Output = Var<Vector<T, M>>;

    fn mul(self, rhs: Var<Vector<T, N>>) -> Var<Vector<T, M>> {
        let op = MatVecMulOp::<T, M, N>::new();

        self.binary_output(&rhs, op)
    }
}

impl<Tn> std::ops::Add<Var<Tn>> for Var<Tn>
where
    Tn: DifferentiableTensor + for<'a> std::ops::Add<&'a Tn, Output = Tn>,
    Tn::T: Differentiable,
{
    type Output = Self;

    fn add(self, rhs: Var<Tn>) -> Self {
        if self.id() == rhs.id() {
            return self * Var::input(Scalar::from(Tn::T::two()));
        }
        let op = AddOp::<Tn>::new();

        self.binary_output(&rhs, op)
    }
}

impl<Tn> std::ops::Sub<Var<Tn>> for Var<Tn>
where
    Tn: DifferentiableTensor + for<'a> std::ops::Add<&'a Tn, Output = Tn>,
    Tn::T: Differentiable,
{
    type Output = Self;

    fn sub(self, rhs: Var<Tn>) -> Self {
        self + (-rhs)
    }
}

impl<Tn> std::ops::Neg for Var<Tn>
where
    Tn: DifferentiableTensor + for<'a> std::ops::Add<&'a Tn, Output = Tn>,
    Tn::T: Differentiable + One,
{
    type Output = Self;

    fn neg(self) -> Self {
        self * Var::input(Scalar::from(-Tn::T::one()))
    }
}

impl<Tn> std::ops::Div<Var<Scalar<Tn::T>>> for Var<Tn>
where
    Tn: DifferentiableTensor,
    Tn::T: Differentiable + One,
{
    type Output = Self;

    fn div(self, other: Var<Scalar<Tn::T>>) -> Self {
        let inv = other.elem_pow(-Tn::T::one());

        self * inv
    }
}

impl<Tn> std::cmp::PartialEq for Var<Tn>
where
    Tn: DifferentiableTensor + PartialEq,
    Tn::T: Differentiable + One,
{
    fn eq(&self, other: &Self) -> bool {
        *self.data() == *other.data()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tensor::Tensor, vector::Vector};
    use proptest::prelude::*;

    #[test]
    fn test_equal() {
        let m: Matrix<f64, 3, 2> = Matrix::from([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let v = Var::input(m);
        let s = v.sum_elems();

        let other = Scalar::from(1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0);
        assert_eq!(&*s.data(), &other);
        assert_eq!(s, Var::param(other.clone()));
        assert_eq!(s, Var::input(other.clone()));
        assert!(!s.is(&Var::param(other)));
        assert!(s.is(&s.clone()));
    }

    proptest! {
        #[test]
        fn test_sum(v in prop::collection::vec(any::<f64>(), 6 * 4)) {
            let x: Matrix<f64, 6, 4> = v.into_iter().collect();
            let v = Var::param(x);
            let s = v.sum_elems();

            s.backward().unwrap();

            assert_eq!(*v.as_param().unwrap().grad().unwrap(), Matrix::<f64, 6, 4>::ones());
        }
    }

    #[test]
    fn test_matvec_mul() {
        let a = Var::param(Matrix::<f64, _, _>::from([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]));
        let x = Var::param(Vector::<f64, _>::from([7.0, 8.0]));
        let y = a.clone() * x.clone();
        let l = y.sum_elems();
        assert_eq!(l.data().val(), 159.0);

        l.backward().unwrap();
        assert_eq!(
            &*a.as_param().unwrap().grad().unwrap(),
            &Matrix::<f64, _, _>::from([[7.0, 8.0], [7.0, 8.0], [7.0, 8.0]])
        );
        assert_eq!(
            &*x.as_param().unwrap().grad().unwrap(),
            &Vector::<f64, _>::from([9.0, 12.0])
        );
    }

    #[test]
    fn test_log() {
        let x: Var<Vector<f64, _>> = Var::param([1.0, 2.0, 3.0, 4.0, 5.0].into());
        let y = x.elem_ln().sum_elems();
        y.backward().unwrap();

        const TOLERANCE: f64 = 1e-3;
        assert!((y.data().val() - 4.7875).abs() < TOLERANCE);
        assert_eq!(
            *x.as_param().unwrap().grad().unwrap(),
            Vector::<f64, _>::from([1.0, 1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0, 1.0 / 5.0])
        );
    }

    #[test]
    fn test_dim_sum() {
        let x = Var::param(Matrix::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
        let y = x.clone().dim_sum::<1>();
        let z = y.sum_elems();
        z.backward().unwrap();
        let grad = x.as_param().unwrap().grad().unwrap().clone();
        assert_eq!(grad, Matrix::<f64, _, _>::ones());
    }

    #[test]
    fn test_bcast_elem_mul() {
        let x = Var::param(Matrix::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
        let y = Var::param(Scalar::from(2.0));
        let z = x.elem_mul(y.clone());
        let l = z.sum_elems();
        l.backward().unwrap();

        assert_eq!(l.data().val(), 42.0);
        assert_eq!(y.as_param().unwrap().grad().unwrap().val(), 21.0);
    }

    #[test]
    fn test_bcast_elem_add() {
        let x = Var::param(Matrix::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
        let y = Var::param(Scalar::from(1.0));
        let z = x.elem_add(y.clone());
        let l = z.sum_elems();
        assert_eq!(l.data().val(), 27.0);

        l.backward().unwrap();
        assert_eq!(y.as_param().unwrap().grad().unwrap().val(), 6.0);
    }

    #[test]
    fn test_bcast_elem_div() {
        let x = Var::param(Matrix::from([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]));
        let y = Var::param(Scalar::from(2.0));
        let z = x.elem_div(y.clone());
        let l = z.sum_elems();
        assert_eq!(l.data().val(), 21.0);

        l.backward().unwrap();
        assert_eq!(y.as_param().unwrap().grad().unwrap().val(), -10.5);
    }
}
