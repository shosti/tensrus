pub trait Numeric:
    num::Signed + Copy + std::fmt::Display + std::ops::MulAssign + std::ops::AddAssign + std::ops::Neg
{
}

impl Numeric for i32 {}
impl Numeric for i64 {}
impl Numeric for f32 {}
impl Numeric for f64 {}
