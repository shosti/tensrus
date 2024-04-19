use crate::numeric::Numeric;
use rand::random;
use std::cell::RefCell;
use std::collections::hash_set::HashSet;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

pub struct Value<T: Numeric> {
    inner: Rc<RefCell<ValueInner<T>>>,
}

struct ValueInner<T: Numeric> {
    id: u64,
    data: T,
    grad: f64,
    backward: Option<Box<dyn FnMut() -> ()>>,
    prev: HashSet<Value<T>>,
    op: String,
}

impl<T: Numeric> Value<T> {
    pub fn new(val: T) -> Self {
        let inner = Rc::new(RefCell::new(ValueInner {
            id: random(),
            data: val,
            grad: 0.0,
            backward: None,
            prev: HashSet::new(),
            op: "".to_string(),
        }));
        Self { inner }
    }

    // pub fn add(&mut self, other: &mut Self) -> Self {
    //     let out = Self {
    //         id: random(),
    //         data: self.data + other.data,
    //         grad: 0.0,
    //         backward: None,
    //         prev: HashSet::from([self, other]),
    //         op: "+".to_string(),
    //     };

    //     out
    // }
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
