use crate::numeric::Numeric;
use rand::random;
use std::cell::RefCell;
use std::collections::hash_set::HashSet;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

pub struct Value<T: Numeric> {
    inner: Rc<RefCell<ValueInner<T>>>,
}

struct ValueInner<T: Numeric> {
    id: u64,
    data: T,
    grad: T,
    backward: Option<Box<dyn FnMut(T, T) -> ()>>,
    prev: HashSet<Value<T>>,
    op: String,
}

impl<T: Numeric> Value<T> {
    pub fn new(val: T) -> Self {
        let inner = Rc::new(RefCell::new(ValueInner {
            id: random(),
            data: val,
            grad: T::zero(),
            backward: None,
            prev: HashSet::new(),
            op: "".to_string(),
        }));
        Self { inner }
    }

    fn new_from_op(data: T, prev: HashSet<Value<T>>, op: String) -> Self {
        let inner = Rc::new(RefCell::new(ValueInner {
            id: random(),
            data,
            grad: T::zero(),
            backward: None,
            prev,
            op,
        }));

        Self { inner }
    }

    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        Self::build_topo(self, &mut topo, &mut visited);

        self.inner.borrow_mut().grad = T::one();
        for val in topo.iter().rev() {
            let grad;
            let data;
            {
                let inner = val.inner.borrow();
                grad = inner.grad;
                data = inner.data;
            }
            if let Some(backward) = &mut val.inner.borrow_mut().backward {
                backward(grad, data);
            }
        }
    }

    fn build_topo(cur: &Value<T>, topo: &mut Vec<Self>, visited: &mut HashSet<u64>) {
        let val = cur.inner.borrow();
        if visited.contains(&val.id) {
            return;
        }

        visited.insert(val.id);
        for child in val.prev.iter() {
            Self::build_topo(child, topo, visited);
        }
        topo.push(cur.clone());
    }

    fn pow(&self, n: T) -> Self {
        let data = self.inner.borrow().data.powf(n);
        let children = HashSet::from([self.clone()]);
        let out = Self::new_from_op(data, children, "^".to_string());

        let self_grad = self.clone();
        let backward = move |grad, _| {
            let mut self_inner = self_grad.inner.borrow_mut();
            self_inner.grad += (n * data.powf(n - T::one())) * grad;
        };
        out.inner.borrow_mut().backward = Some(Box::new(backward));

        out
    }

    pub fn relu(&self) -> Self {
        let data = self.inner.borrow().data;
        let outval = if data.is_sign_negative() {
            T::zero()
        } else {
            data
        };
        let children = HashSet::from([self.clone()]);
        let out = Self::new_from_op(outval, children, "ReLU".to_string());

        let self_grad = self.clone();
        let backward = move |grad, data: T| {
            let mut self_inner = self_grad.inner.borrow_mut();
            self_inner.grad += if data.is_sign_positive() {
                grad
            } else {
                T::zero()
            };
        };
        out.inner.borrow_mut().backward = Some(Box::new(backward));

        out
    }

    pub fn zero_grad(&self) {
        self.inner.borrow_mut().grad = T::zero();
    }
}

impl<T: Numeric> Add for Value<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        let data = self.inner.borrow().data + other.inner.borrow().data;
        let children = HashSet::from([self.clone(), other.clone()]);
        let out = Self::new_from_op(data, children, "+".to_string());

        let self_grad = self.clone();
        let other_grad = other.clone();
        let backward = move |grad, _| {
            let mut self_inner = self_grad.inner.borrow_mut();
            let mut other_inner = other_grad.inner.borrow_mut();
            self_inner.grad += grad;
            other_inner.grad += grad;
        };
        out.inner.borrow_mut().backward = Some(Box::new(backward));

        out
    }
}

impl<T: Numeric> Sub for Value<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self + (-other)
    }
}

impl<T: Numeric> Div for Value<T> {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        let inv = other.pow(-T::one());

        self * inv
    }
}

impl<T: Numeric> Mul for Value<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        let data = self.inner.borrow().data * other.inner.borrow().data;
        let children = HashSet::from([self.clone(), other.clone()]);
        let out = Self::new_from_op(data, children, "*".to_string());

        let self_grad = self.clone();
        let other_grad = other.clone();
        let backward = move |grad, _| {
            let mut self_inner = self_grad.inner.borrow_mut();
            let mut other_inner = other_grad.inner.borrow_mut();
            self_inner.grad += other_inner.grad * grad;
            other_inner.grad += self_inner.grad * grad;
        };
        out.inner.borrow_mut().backward = Some(Box::new(backward));

        out
    }
}

impl<T: Numeric> Neg for Value<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * Value::new(-T::one())
    }
}

impl<T: Numeric> Clone for Value<T> {
    fn clone(&self) -> Self {
        Value {
            inner: self.inner.clone(),
        }
    }
}

impl<T: Numeric> PartialEq for Value<T> {
    fn eq(&self, other: &Self) -> bool {
        self.inner.borrow().id == other.inner.borrow().id
    }
}

impl<T: Numeric> Eq for Value<T> {}

impl<T: Numeric> Hash for Value<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.borrow().id.hash(state);
    }
}

impl<T: Numeric> std::iter::Sum for Value<T> {
    fn sum<I: Iterator<Item = Value<T>>>(mut iter: I) -> Self {
        let first = iter.next();
        if first.is_none() {
            return Value::new(T::zero());
        }
        let mut res = first.unwrap();

        while let Some(next) = iter.next() {
            res = res + next;
        }

        res
    }
}

impl<T: Numeric> std::fmt::Debug for Value<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let val = self.inner.borrow();
        write!(
            f,
            "Value(id={}, data={}, grad={}, op={})",
            val.id, val.data, val.grad, val.op
        )
    }
}
