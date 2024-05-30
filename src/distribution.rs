use rand::Rng;
use rand_distr::Distribution;

use crate::{numeric::Numeric, vector::NormalizedVector};

#[derive(Debug)]
pub struct Multinomial<'a, T: Numeric, const N: usize> {
    v: &'a NormalizedVector<T, N>,
}

impl<'a, T: Numeric, const N: usize> From<&'a NormalizedVector<T, N>> for Multinomial<'a, T, N> {
    fn from(v: &'a NormalizedVector<T, N>) -> Self {
        Self { v }
    }
}

impl<'a, T: Numeric, const N: usize> Distribution<usize> for Multinomial<'a, T, N> {
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

        N - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::Vector;
    use rand::thread_rng;

    #[test]
    fn test_sample() {
        let mut rng = thread_rng();

        let all_zeros_v: NormalizedVector<f64, _> = Vector::from([1.0, 0.0, 0.0]).normalize();
        let all_zeros = Multinomial::from(&all_zeros_v);
        for _ in 0..10 {
            assert_eq!(all_zeros.sample(&mut rng), 0);
        }

        let all_ones_v: NormalizedVector<f64, _> = Vector::from([0.0, 1.0, 0.0]).normalize();
        let all_ones = Multinomial::from(&all_ones_v);
        for _ in 0..10 {
            assert_eq!(all_ones.sample(&mut rng), 1);
        }

        let all_twos_v: NormalizedVector<f64, _> = Vector::from([0.0, 0.0, 1.0]).normalize();
        let all_twos = Multinomial::from(&all_twos_v);
        for _ in 0..10 {
            assert_eq!(all_twos.sample(&mut rng), 2);
        }

        let one_or_two_v: NormalizedVector<f64, _> = Vector::from([0.0, 1.0, 1.0]).normalize();
        let one_or_two = Multinomial::from(&one_or_two_v);
        for _ in 0..10 {
            let s = one_or_two.sample(&mut rng);
            assert!(s == 1 || s == 2);
        }
    }
}
