pub trait Numeric:
    num::Float
    + Copy
    + std::fmt::Display
    + std::ops::MulAssign
    + std::ops::AddAssign
    + rand::distributions::uniform::SampleUniform
{
}

impl Numeric for f32 {}
impl Numeric for f64 {}
