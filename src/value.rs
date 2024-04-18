use crate::numeric::Numeric;
use std::collections::hash_set::HashSet;

pub struct Value<T: Numeric> {
    data: T,
    grad: f64,
    backward: Option<Box<dyn FnMut() -> ()>>,
    prev: HashSet<Value<T>>,
    op: String,
}

impl<T: Numeric> Value<T> {
    pub fn new(val: T) -> Self {
        Self {
            data: val,
            grad: 0.0,
            backward: None,
            prev: HashSet::new(),
            op: "".to_string(),
        }
    }
}

impl<T: Numeric> std::fmt::Debug for Value<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Value(data={}, grad={}, op={})",
            self.data, self.grad, self.op
        )
    }
}
