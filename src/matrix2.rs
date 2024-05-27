use crate::{
    numeric::Numeric,
    shape::Shape,
    storage::{Layout, Storage},
    tensor2::Tensor2,
};

pub const fn matrix_shape(m: usize, n: usize) -> Shape {
    [m, n, 0, 0, 0, 0]
}

#[derive(Tensor2, Debug, Clone)]
#[tensor_rank = 2]
#[tensor_shape = "matrix_shape(M, N)"]
pub struct Matrix2<T: Numeric, const M: usize, const N: usize> {
    storage: Storage<T>,
    layout: Layout,
}

impl<T: Numeric, const M: usize, const N: usize> Matrix2<T, M, N> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_fn() {
        let x: Matrix2<f64, 2, 3> = Matrix2::from_fn(|_| 7.0);
        let y = x.map(|_, x| x * 2.0);
        assert_eq!(y[&[1, 1]], 14.0);
    }
}
