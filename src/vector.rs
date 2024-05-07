use crate::generic_tensor::GenericTensor;
use crate::numeric::Numeric;
use crate::tensor::{num_elems, Tensor, TensorShape};
use num::ToPrimitive;

pub const fn vector_shape(n: usize) -> TensorShape {
    [n, 0, 0, 0, 0]
}

#[derive(Tensor, PartialEq, Debug)]
pub struct Vector<T: Numeric, const N: usize>(pub(crate) GenericTensor<T, 1, { vector_shape(N) }>)
where
    [(); num_elems(1, vector_shape(N))]:;

impl<T: Numeric, const N: usize> Vector<T, N>
where
    [(); num_elems(1, vector_shape(N))]:,
{
    pub fn dot(&self, other: &Self) -> T {
        let mut res = T::zero();
        for i in 0..N {
            let idx = [i];
            res += self.get(idx) * other.get(idx);
        }

        res
    }
}

impl<T: Numeric, const N: usize, F: ToPrimitive> From<[F; N]> for Vector<T, N>
where
    [(); num_elems(1, vector_shape(N))]:,
{
    fn from(arr: [F; N]) -> Self {
        let t: GenericTensor<T, 1, { vector_shape(N) }> = arr.into_iter().collect();
        Self(t)
    }
}

impl<T: Numeric, const N: usize> From<Vector<T, N>> for GenericTensor<T, 1, { vector_shape(N) }>
where
    [(); num_elems(1, vector_shape(N))]:,
{
    fn from(val: Vector<T, N>) -> Self {
        val.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basics() {
        let a: Vector<f64, 5> = Vector::from([1, 2, 3, 4, 5]);

        assert_eq!(a.get([3]), 4.0);
    }

    #[test]
    fn test_from_fn() {
        let a: Vector<_, 4> = Vector::from_fn(|idx| idx[0] as f32 * 2.0);

        assert_eq!(a, Vector::from([0, 2, 4, 6]));
    }

    #[test]
    fn test_dot_product() {
        let a: Vector<f64, _> = Vector::from([1, 2, 3]);
        let b: Vector<f64, _> = Vector::from([4, 5, 6]);

        assert_eq!(a.dot(&b), 32.0);
    }
}
