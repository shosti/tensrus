use crate::numeric::Numeric;
use rand::random;
use std::collections::hash_set::HashSet;
use std::hash::{Hash, Hasher};

pub struct Value<T: Numeric> {
    id: u64,
    data: T,
    grad: f64,
    backward: Option<Box<dyn FnMut() -> ()>>,
    prev: HashSet<Value<T>>,
    op: String,
}

impl<T: Numeric> Value<T> {
    pub fn new(val: T) -> Self {
        Self {
            id: random(),
            data: val,
            grad: 0.0,
            backward: None,
            prev: HashSet::new(),
            op: "".to_string(),
        }
    }
}

impl<T: Numeric> PartialEq for Value<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T: Numeric> Eq for Value<T> {}

impl<T: Numeric> Hash for Value<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl<T: Numeric> std::fmt::Debug for Value<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Value(id={}, data={}, grad={}, op={})",
            self.id, self.data, self.grad, self.op
        )
    }
}
