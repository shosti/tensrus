use crate::{
    shape::{self, Shape},
    storage::TensorStorage,
    tensor::{ShapedTensor, Tensor},
    type_assert::{Assert, IsTrue},
    view::View,
};

pub const fn broadcast_compat(r_src: usize, s_src: Shape, r_dest: usize, s_dest: Shape) -> bool {
    assert!(r_dest >= r_src, "cannot broadcast to lower dimension");

    let s_src_n = broadcast_normalize(s_src, r_src, r_dest);

    let mut i = 0;
    while i < r_dest {
        if s_src_n[i] != 1 && s_src_n[i] != s_dest[i] {
            return false;
        }
        i += 1;
    }

    true
}

pub const fn broadcast_normalize(s: Shape, r_src: usize, r_dest: usize) -> Shape {
    assert!(r_dest >= r_src, "cannot broadcast to a lower dimension");
    assert!(
        r_src <= shape::MAX_DIMS && r_dest <= shape::MAX_DIMS,
        "cannot broadcast to a dimension higher than the max"
    );

    let r_diff = r_dest - r_src;
    let mut ret = [0; shape::MAX_DIMS];

    let mut i = 0;
    while i < r_diff {
        ret[i] = 1;
        i += 1;
    }
    while i < r_dest {
        ret[i] = s[i - r_diff];
        i += 1;
    }

    ret
}

pub trait Broadcastable<T>: ShapedTensor + TensorStorage<T> {
    fn broadcast<Dest>(&self) -> View<Dest>
    where
        Dest: Tensor<T = T> + ShapedTensor,
        Assert<{ broadcast_compat(Self::R, Self::S, Dest::R, Dest::S) }>: IsTrue,
    {
        let layout = self.layout();
        let t = Box::new(move |dest_idx: Dest::Idx| {
            let idx: &[usize] = dest_idx.as_ref();
            let s_normalized = broadcast_normalize(Self::S, Self::R, Dest::R);

            let mut src_idx = [0; Self::R];
            let mut dim = 0;
            for i in 0..Dest::R {
                if s_normalized[i] == 1 && Dest::S[i] != 1 {
                    continue;
                }
                src_idx[dim] = idx[i];
                dim += 1;
            }

            crate::storage::storage_idx_gen(Self::R, &src_idx, Self::S, layout).unwrap()
        });
        View::with_translation(self.storage(), layout, t)
    }
}

#[cfg(test)]
pub mod tests {
    use crate::{generic_tensor::GenericTensor, matrix::Matrix, shape::MAX_DIMS, vector::Vector};

    use super::*;

    #[test]
    fn test_broadcast_normalize() {
        assert_eq!(
            broadcast_normalize([7, 2, 3, 0, 0, 0], 3, 3),
            [7, 2, 3, 0, 0, 0],
        );

        assert_eq!(
            broadcast_normalize([7, 2, 3, 0, 0, 0], 3, 6),
            [1, 1, 1, 7, 2, 3],
        );
    }

    #[test]
    fn test_broadcast_compat() {
        let s = [1000, 256, 256, 256, 0, 0];
        let r = 4;

        assert!(broadcast_compat(4, [1000, 256, 1, 256, 0, 0], r, s));
        assert!(broadcast_compat(4, [1000, 1, 256, 256, 0, 0], r, s));
        assert!(broadcast_compat(2, [256, 1, 0, 0, 0, 0], r, s));
        assert!(!broadcast_compat(3, [1000, 256, 256, 0, 0, 0], r, s));
        assert!(!broadcast_compat(r, s, 4, [1000, 256, 1, 256, 0, 0]));
    }

    #[test]
    fn test_broadcast() {
        let v = Vector::<f64, _>::from([1, 2, 3]);
        let m: Matrix<_, 3, 3> = v.broadcast().into();
        assert_eq!(
            m,
            Matrix::<f64, _, _>::from([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        );

        let t: GenericTensor<_, 3, { [3; MAX_DIMS] }> = v.broadcast().into();
        assert_eq!(
            t,
            [1, 2, 3]
                .into_iter()
                .cycle()
                .collect::<GenericTensor<f64, 3, { [3; MAX_DIMS] }>>()
        );
    }
}
