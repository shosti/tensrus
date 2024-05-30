use rand::Rng;
use rand_distr::Distribution;

use crate::{numeric::Numeric, vector::Vector};

#[derive(Debug)]
pub struct Multinomial<T: Numeric, const N: usize> {
    v: Vector<T, N>,
}

impl<T: Numeric, const N: usize> From<Vector<T, N>> for Multinomial<T, N> {
    fn from(v: Vector<T, N>) -> Self {
        Self { v: v.normalize() }
    }
}

impl<T: Numeric, const N: usize> Distribution<usize> for Multinomial<T, N> {
    fn sample<R>(&self, rng: &mut R) -> usize
    where
        R: Rng + ?Sized,
    {
        let n = rng.gen_range(T::zero()..=T::one());
        let mut so_far = T::zero();

        for i in 0..N {
            let cur = self.v[&[i]];
            if so_far + cur >= n {
                return i;
            }
            so_far += cur;
        }

        return N - 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn test_sample() {
        let mut rng = thread_rng();

        let all_zeros: Multinomial<f64, _> = Vector::from([1.0, 0.0, 0.0]).into();
        for _ in 0..10 {
            assert_eq!(all_zeros.sample(&mut rng), 0);
        }

        let all_ones: Multinomial<f64, _> = Vector::from([0.0, 1.0, 0.0]).into();
        for _ in 0..10 {
            assert_eq!(all_ones.sample(&mut rng), 1);
        }

        let all_twos: Multinomial<f64, _> = Vector::from([0.0, 0.0, 1.0]).into();
        for _ in 0..10 {
            assert_eq!(all_twos.sample(&mut rng), 2);
        }

        let one_or_two: Multinomial<f64, _> = Vector::from([0.0, 1.0, 1.0]).into();
        for _ in 0..10 {
            let s = one_or_two.sample(&mut rng);
            assert!(s == 1 || s == 2);
        }
    }
}
