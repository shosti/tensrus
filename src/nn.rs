use rand::thread_rng;
use rand_distr::{Distribution, StandardNormal};

use crate::numeric::Numeric;
use crate::scalar::Scalar;
use crate::tensor::Tensor;
use crate::var::Var;

pub trait Module<T: Numeric> {
    fn parameters(&self) -> Vec<Var<Scalar<T>>>;
}

pub struct Neuron<T: Numeric> {
    w: Vec<Var<Scalar<T>>>,
    b: Var<Scalar<T>>,
    nonlin: bool,
}

impl<T: Numeric> Neuron<T> {
    pub fn new(nin: usize, nonlin: bool) -> Self
    where
        StandardNormal: Distribution<T>,
    {
        let mut w = Vec::new();
        for _ in 0..nin {
            w.push(Var::new(Scalar::randn(&mut thread_rng())));
        }

        Self {
            w,
            b: Var::from(T::zero()),
            nonlin,
        }
    }

    pub fn call(&self, x: &[Var<Scalar<T>>]) -> Var<Scalar<T>> {
        let wx: Var<Scalar<T>> = std::iter::zip(self.w.clone(), x)
            .map(|(wi, xi)| (wi * xi.clone()))
            .sum();
        let act = wx + self.b.clone();
        if self.nonlin {
            act.relu()
        } else {
            act
        }
    }
}

impl<T: Numeric> Module<T> for Neuron<T> {
    fn parameters(&self) -> Vec<Var<Scalar<T>>> {
        let mut p = self.w.clone();
        p.push(self.b.clone());

        p
    }
}

pub struct Layer<T: Numeric> {
    neurons: Vec<Neuron<T>>,
}

impl<T: Numeric> Layer<T> {
    pub fn new(nin: usize, nout: usize, nonlin: bool) -> Self
    where
        StandardNormal: Distribution<T>,
    {
        let mut neurons = Vec::new();
        for _ in 0..nout {
            neurons.push(Neuron::new(nin, nonlin));
        }

        Self { neurons }
    }

    pub fn call(&self, x: &[Var<Scalar<T>>]) -> Vec<Var<Scalar<T>>> {
        let mut out = Vec::new();
        for n in self.neurons.iter() {
            out.push(n.call(x));
        }

        out
    }
}

impl<T: Numeric> Module<T> for Layer<T> {
    fn parameters(&self) -> Vec<Var<Scalar<T>>> {
        let mut res = Vec::new();
        for n in self.neurons.iter() {
            for p in n.parameters().iter() {
                res.push(p.clone());
            }
        }

        res
    }
}

pub struct MLP<T: Numeric> {
    layers: Vec<Layer<T>>,
}

impl<T: Numeric> MLP<T> {
    pub fn new(nin: usize, nouts: Vec<usize>) -> Self
    where
        StandardNormal: Distribution<T>,
    {
        let mut sizes = vec![nin];
        sizes.extend(nouts.iter());
        let mut layers = Vec::new();
        for i in 0..(nouts.len()) {
            let nonlinear = i != nouts.len() - 1;
            let layer = Layer::new(sizes[i], sizes[i + 1], nonlinear);
            layers.push(layer);
        }

        MLP { layers }
    }

    pub fn call(&self, x: Vec<Var<Scalar<T>>>) -> Vec<Var<Scalar<T>>> {
        let mut res = x;
        for layer in self.layers.iter() {
            res = layer.call(&res);
        }

        res
    }

    pub fn loss(&self, ys: &[Var<Scalar<T>>], ypred: &[Var<Scalar<T>>]) -> Var<Scalar<T>> {
        std::iter::zip(ys.iter(), ypred.iter())
            .map(|(ygt, yout)| (yout.clone() - ygt.clone()).elem_pow(T::from(2.0).unwrap()))
            .sum()
    }
}

impl<T: Numeric> Module<T> for MLP<T> {
    fn parameters(&self) -> Vec<Var<Scalar<T>>> {
        let mut res = Vec::new();
        for layer in self.layers.iter() {
            for p in layer.parameters().iter() {
                res.push(p.clone());
            }
        }

        res
    }
}
