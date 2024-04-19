use crate::numeric::Numeric;
use crate::value::Value;
use rand::distributions::{Distribution, Uniform};

pub trait Module<T: Numeric + 'static> {
    fn zero_grad(&self) {
        for p in self.parameters().iter() {
            p.zero_grad();
        }
    }
    fn parameters(&self) -> Vec<Value<T>>;
}

pub struct Neuron<T: Numeric> {
    w: Vec<Value<T>>,
    b: Value<T>,
    nonlin: bool,
}

impl<T: Numeric + 'static> Neuron<T> {
    pub fn new(nin: usize, nonlin: bool) -> Self {
        let mut w = Vec::new();
        let between = Uniform::from(-T::one()..T::one());
        let mut rng = rand::thread_rng();
        for _ in 0..nin {
            w.push(Value::new(between.sample(&mut rng)));
        }

        Self {
            w,
            b: Value::new(T::zero()),
            nonlin,
        }
    }

    pub fn call(&self, x: Vec<Value<T>>) -> Value<T> {
        let act: Value<T> = std::iter::zip(self.w.clone(), x).map(|(wi, xi)| wi * xi).sum();
        if self.nonlin {
            act.relu()
        } else {
            act
        }
    }
}

impl<T: Numeric + 'static> Module<T> for Neuron<T> {
    fn parameters(&self) -> Vec<Value<T>> {
        let mut p = self.w.clone();
        p.push(self.b.clone());

        p
    }
}
