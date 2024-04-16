pub trait Tensor<T: Numeric>: Index<Vec>{
}

#[derive(Debug)]
pub struct Value<T: Numeric> {
    mem: T
}

impl<T: Numeric> Value {
    pub fn new(val: T) -> Self {
        Value {
            mem: val
        }
    }
}

#[derive(Debug)]
pub struct Vector<T: Numeric, const N: usize> {
    mem: [N; T]
}

#[derive(Debug)]
pub struct Matrix<T: Numeric, const N: usize, const M: usize> {
    m: [N*M; T]
}
