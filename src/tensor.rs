use std::ops::MulAssign;
use num::Num;

pub trait Tensor<T: Num + Copy, const R: usize>: MulAssign<T> {
    fn rank(&self) -> usize {
        R
    }

    fn shape(&self) -> [usize; R];
}
