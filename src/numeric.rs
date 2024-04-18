use num::Num;

pub trait Numeric:
    Num + Copy + std::fmt::Display + std::ops::MulAssign + std::ops::AddAssign
{
}

impl Numeric for i32 {}
impl Numeric for i64 {}
impl Numeric for f32 {}
impl Numeric for f64 {}
impl Numeric for usize {}
