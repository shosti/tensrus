pub mod numeric;
pub mod tensor;
pub mod matrix;
pub mod vector;
pub mod scalar;
pub mod value;

pub use tensor::Tensor;
use value::Value;

fn main() {
    let x = Value::new(3.0);
    let y = Value::new(2.5);
    let z = x.add(&y);

    println!("Z: {:#?}", z);
}
