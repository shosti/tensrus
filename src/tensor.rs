use num::Num;

pub trait Tensor<T: Num, const DIM: usize> {
    fn dim(&self) -> usize {
        DIM
    }

    fn shape(&self) -> [usize; DIM];
}

#[derive(Debug)]
pub struct Value<T: Num> {
    mem: T,
}

impl<T: Num> From<T> for Value<T> {
    fn from(val: T) -> Self {
        Value { mem: val }
    }
}

impl<T: Num> Tensor<T, 0> for Value<T> {
    fn shape(&self) -> [usize; 0] {
        []
    }
}

#[derive(Debug)]
pub struct Vector<T: Num, const N: usize> {
    shape: (usize,),
    mem: [T; N],
}

impl<T: Num, const N: usize> From<[T; N]> for Vector<T, N> {
    fn from(vals: [T; N]) -> Self {
        Vector {
            shape: (N,),
            mem: vals,
        }
    }
}

impl<T: Num, const N: usize> Tensor<T, 1> for Vector<T, N> {
    fn shape(&self) -> [usize; 1] {
        [self.shape.0]
    }
}

#[derive(Debug)]
pub struct Matrix<T: Num, const N: usize, const M: usize>
where
    [(); N * M]:,
{
    shape: (usize, usize),
    mem: [T; N * M],
}

impl<T: Num + Copy, const N: usize, const M: usize> From<[[T; N]; M]> for Matrix<T, N, M>
where
    [(); N * M]:,
{
    fn from(vals: [[T; N]; M]) -> Self {
        let mut ret: Matrix<T, N, M> = Matrix {
            shape: (N, M),
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

impl<T: Num, const N: usize, const M: usize> Tensor<T, 2> for Matrix<T, N, M> where [(); N * M]: {
    fn shape(&self) -> [usize; 2] {
        [self.shape.0, self.shape.1]
    }
}
