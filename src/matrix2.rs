use std::ops::Index;

use crate::{
    matrix::matrix_shape,
    numeric::Numeric,
    storage::{nth_idx, storage_idx, Storage},
    tensor::{num_elems, Shape},
    tensor2::{Layout, Tensor2},
};

const RANK: usize = 2;

#[derive(Debug, Clone)]
pub struct Matrix2<T: Numeric, const M: usize, const N: usize> {
    storage: Storage<T>,
    layout: Layout,
}

impl<T: Numeric, const M: usize, const N: usize> Matrix2<T, M, N> {
    fn shape() -> Shape {
        matrix_shape(M, N)
    }
}

impl<T: Numeric, const M: usize, const N: usize> Tensor2 for Matrix2<T, M, N> {
    type T = T;
    type Idx = [usize; 2];

    // Supplied methods
    fn from_fn(f: impl Fn(&Self::Idx) -> Self::T) -> Self {
        let mut v = Vec::with_capacity(Self::num_elems());
        let layout = Layout::Normal;

        for i in 0..Self::num_elems() {
            let idx = nth_idx(i, Self::shape(), layout).unwrap();
            v.push(f(&idx));
        }

        Self {
            storage: v.into(),
            layout,
        }
    }
    fn map(mut self, f: impl Fn(&Self::Idx, Self::T) -> Self::T) -> Self {
        let mut next_idx = Some(Self::default_idx());
        while let Some(idx) = next_idx {
            let i = storage_idx(&idx, Self::shape(), self.layout).unwrap();
            self.storage[i] = f(&idx, self.storage[i]);
            next_idx = self.next_idx(&idx);
        }

        Self {
            storage: self.storage,
            layout: self.layout,
        }
    }
    fn num_elems() -> usize {
        num_elems(RANK, matrix_shape(M, N))
    }
    fn default_idx() -> Self::Idx {
        [0; RANK]
    }
    fn next_idx(&self, idx: &Self::Idx) -> Option<Self::Idx> {
        let i = storage_idx(idx, Self::shape(), self.layout).ok()?;
        if i >= Self::num_elems() - 1 {
            return None;
        }

        nth_idx(i + 1, Self::shape(), self.layout).ok()
    }
}

impl<T: Numeric, const M: usize, const N: usize> Index<&[usize; RANK]> for Matrix2<T, M, N> {
    type Output = T;

    fn index(&self, idx: &[usize; RANK]) -> &Self::Output {
        let i = storage_idx::<RANK>(idx, Self::shape(), self.layout).unwrap();
        self.storage.index(i)
    }
}

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
