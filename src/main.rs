pub mod numeric;
pub mod tensor;
pub mod matrix;
pub mod vector;
pub mod scalar;
pub mod value;

pub use tensor::Tensor;
use value::Value;

fn main() {
    let mut x = Value::new(3.0);
    let mut y = Value::new(2.5);
    let z = x.add(&mut y);

    println!("Z: {:#?}", z);
}
