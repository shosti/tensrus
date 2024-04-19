pub mod matrix;
pub mod numeric;
pub mod scalar;
pub mod tensor;
pub mod value;
pub mod vector;
pub mod nn;

pub use tensor::Tensor;
use value::Value;

fn main() {
    let x = Value::new(3.0);
    let y = Value::new(2.5);
    let z = x + y;
    z.backward();

    println!("Z: {:#?}", z);
}
