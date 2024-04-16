use num::Num;

pub trait Tensor<T: Num> {}

#[derive(Debug)]
pub struct Value<T: Num> {
    mem: T,
}

impl<T: Num> From<T> for Value<T> {
    fn from(val: T) -> Self {
        Value { mem: val }
    }
}

#[derive(Debug)]
pub struct Vector<T: Num, const N: usize> {
    dim: (usize,),
    mem: [T; N],
}

impl<T: Num, const N: usize> From<[T; N]> for Vector<T, N> {
    fn from(vals: [T; N]) -> Self {
        Vector {
            dim: (N,),
            mem: vals,
        }
    }
}

#[derive(Debug)]
pub struct Matrix<T: Num, const N: usize, const M: usize>
where
    [(); N * M]:,
{
    dim: (usize, usize),
    mem: [T; N * M],
}

impl<T: Num + Copy, const N: usize, const M: usize> From<[[T; N]; M]> for Matrix<T, N, M>
where
    [(); N * M]:,
{
    fn from(vals: [[T; N]; M]) -> Self {
        let mut ret: Matrix<T, N, M> = Matrix {
            dim: (N, M),
            mem: std::array::from_fn(|_| T::zero()),
        };
        for i in 0..M {
            for j in 0..N {
                ret.mem[(i * N) + j] = vals[i][j];
            }
        }

        ret
    }
}
