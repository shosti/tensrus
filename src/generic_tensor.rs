use crate::{
    numeric::Numeric,
    shape::{subtensor_shape, transpose_shape, Shape},
    storage::{num_elems, storage_idx, IndexError, Layout, Storage},
    tensor::Tensor,
    type_assert::{Assert, IsTrue},
};
use num::ToPrimitive;

#[derive(Tensor, Debug, Clone)]
#[tensor_rank = "R"]
#[tensor_shape = "S"]
pub struct GenericTensor<T: Numeric, const R: usize, const S: Shape> {
    storage: Storage<T>,
    layout: Layout,
}

impl<T: Numeric, const R: usize, const S: Shape, U: ToPrimitive> From<[U; num_elems(R, S)]>
    for GenericTensor<T, R, S>
{
    fn from(arr: [U; num_elems(R, S)]) -> Self {
        let vals = arr.into_iter().map(|v| T::from(v).unwrap()).collect();
        Self {
            storage: vals,
            layout: Layout::default(),
        }
    }
}

impl<T: Numeric, const R: usize, const S: Shape> GenericTensor<T, R, S> {
    pub fn transpose(self) -> GenericTensor<T, R, { transpose_shape(R, S) }>
    where
        Assert<{ R >= 2 }>: IsTrue,
    {
        GenericTensor {
            storage: self.storage,
            layout: self.layout.transpose(),
        }
    }

    pub fn reshape<const R2: usize, const S2: Shape>(self) -> GenericTensor<T, R2, S2>
    where
        Assert<{ num_elems(R, S) == num_elems(R2, S2) }>: IsTrue,
    {
        GenericTensor {
            storage: self.storage,
            layout: self.layout,
        }
    }

    pub fn subtensor(
        &self,
        i: usize,
    ) -> Result<GenericTensor<T, { R - 1 }, { subtensor_shape(R, S) }>, IndexError> {
        if i >= S[0] {
            return Err(IndexError {});
        }

        let out: GenericTensor<T, { R - 1 }, { subtensor_shape(R, S) }> =
            GenericTensor::from_fn(|idx| {
                let mut self_idx = [i; R];
                self_idx[1..R].copy_from_slice(&idx[..(R - 1)]);
                self.storage[storage_idx(&self_idx, S, self.layout).unwrap()]
            });
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equal() {
        let a: GenericTensor<f64, 0, { [0; 6] }> = GenericTensor::from([1]);
        let b: GenericTensor<f64, 0, { [0; 6] }> = GenericTensor::from([2]);
        assert_ne!(a, b);

        let x: GenericTensor<f64, 2, { [2; 6] }> = GenericTensor::from([1, 2, 3, 4]);
        let y: GenericTensor<f64, 2, { [2; 6] }> = GenericTensor::from([1, 2, 3, 5]);
        assert_ne!(x, y);
    }

    #[test]
    fn test_from_iterator() {
        let xs: [i64; 3] = [1, 2, 3];
        let iter = xs.iter().cycle().copied();

        let t1: GenericTensor<f64, 0, { [0; 6] }> = iter.clone().collect();
        assert_eq!(t1, GenericTensor::<f64, 0, { [0; 6] }>::from([1.0]));

        let t2: GenericTensor<f64, 2, { [4, 2, 0, 0, 0, 0] }> = iter.clone().collect();
        assert_eq!(
            t2,
            GenericTensor::<f64, 2, { [4, 2, 0, 0, 0, 0] }>::from([
                1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0
            ])
        );

        let t3: GenericTensor<f64, 2, { [4, 2, 0, 0, 0, 0] }> = xs.iter().copied().collect();
        assert_eq!(
            t3,
            GenericTensor::<f64, 2, { [4, 2, 0, 0, 0, 0] }>::from([
                1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ])
        );
    }

    #[test]
    #[allow(clippy::zero_prefixed_literal)]
    fn test_from_fn() {
        let f = |idx: &[usize; 3]| {
            let [i, j, k] = *idx;
            let s = format!("{}{}{}", i, j, k);
            s.parse().unwrap()
        };
        let t1: GenericTensor<f64, 3, { [2; 6] }> = GenericTensor::from_fn(f);
        assert_eq!(
            t1,
            GenericTensor::<f64, 3, { [2; 6] }>::from([000, 001, 010, 011, 100, 101, 110, 111]),
        );

        let t2: GenericTensor<f64, 3, { [1, 2, 3, 0, 0, 0] }> = GenericTensor::from_fn(f);
        assert_eq!(
            t2,
            GenericTensor::<f64, 3, { [1, 2, 3, 0, 0, 0] }>::from([000, 001, 002, 010, 011, 012]),
        );

        let t3: GenericTensor<f64, 3, { [3, 2, 1, 0, 0, 0] }> = GenericTensor::from_fn(f);
        assert_eq!(
            t3,
            GenericTensor::<f64, 3, { [3, 2, 1, 0, 0, 0] }>::from([000, 010, 100, 110, 200, 210]),
        );

        let t4: GenericTensor<f64, 3, { [2, 3, 1, 0, 0, 0] }> = GenericTensor::from_fn(f);
        assert_eq!(
            t4,
            GenericTensor::<f64, 3, { [2, 3, 1, 0, 0, 0] }>::from([000, 010, 020, 100, 110, 120]),
        );
    }

    #[test]
    fn test_math() {
        let mut x: GenericTensor<f64, 3, { [1, 2, 2, 0, 0, 0] }> =
            GenericTensor::from([1, 2, 3, 4]);
        let y: GenericTensor<f64, 3, { [1, 2, 2, 0, 0, 0] }> = GenericTensor::from([5, 6, 7, 8]);
        let a: GenericTensor<f64, 3, { [1, 2, 2, 0, 0, 0] }> =
            GenericTensor::from([6, 8, 10, 12]);

        assert_eq!(x.clone() + &y, a);

        x = x + &y;
        assert_eq!(x.clone(), a);

        let b: GenericTensor<f64, 3, { [1, 2, 2, 0, 0, 0] }> =
            GenericTensor::from([12, 16, 20, 24]);
        assert_eq!(x.clone() * 2.0, b);

        x = x * 2.0;
        assert_eq!(x, b);
    }

    #[test]
    fn test_to_iter() {
        let t: GenericTensor<f64, 2, { [2; 6] }> = (0..4).collect();
        let vals: Vec<f64> = t.iter().values().collect();
        assert_eq!(vals, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_map() {
        let t: GenericTensor<f64, 2, { [2; 6] }> = GenericTensor::from([1, 2, 3, 4]);
        let u = t.map(|_, val| val * 2.0);

        let want: GenericTensor<f64, 2, { [2; 6] }> = GenericTensor::from([2, 4, 6, 8]);
        assert_eq!(u, want);
    }

    #[test]
    fn test_subtensor() {
        let t3: GenericTensor<f64, 3, { [2, 3, 4, 0, 0, 0] }> = (1..25).collect();

        let t2 = t3.subtensor(1).unwrap();
        let t2_expected: GenericTensor<f64, 2, { [3, 4, 0, 0, 0, 0] }> = (13..25).collect();
        assert_eq!(t2, t2_expected);
        assert_eq!(t3.subtensor(2), Err(IndexError {}));

        let t1 = t2.subtensor(1).unwrap();
        let t1_expected: GenericTensor<f64, 1, { [4, 0, 0, 0, 0, 0] }> =
            GenericTensor::from([17, 18, 19, 20]);
        assert_eq!(t1, t1_expected);

        let t0 = t1.subtensor(1).unwrap();
        let t0_expected: GenericTensor<f64, 0, { [0; 6] }> = GenericTensor::from([18]);
        assert_eq!(t0, t0_expected);
    }

    #[test]
    #[rustfmt::skip]
    fn test_transpose() {
        let t: GenericTensor<f64, 2, { [3, 2, 0, 0, 0, 0] }> = GenericTensor::from([
            1, 2,
            3, 4,
            5, 6,
        ]);
        let t2 = t.transpose().map(|_, v| v + 1.0);
        let want: GenericTensor<f64, 2, { [2, 3, 0, 0, 0, 0] }> = GenericTensor::from([
            2, 4, 6,
            3, 5, 7,
        ]);

        assert_eq!(t2, want);
    }

    #[test]
    fn test_reshape() {
        let t = GenericTensor::<f64, 2, { [3, 2, 0, 0, 0, 0] }>::from([1, 2, 3, 4, 5, 6]);
        let t2 = t.clone().reshape::<2, { [2, 3, 0, 0, 0, 0] }>();
        let t3 = t.clone().reshape::<1, { [6, 0, 0, 0, 0, 0] }>();

        assert_eq!(t[&[1, 0]], t2[&[0, 2]]);
        assert_eq!(t[&[2, 1]], t3[&[5]]);
    }
}
