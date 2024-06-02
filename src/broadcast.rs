use crate::{shape::{self, Shape}, tensor::Tensor};

pub const fn broadcast_compat(r1: usize, s1: Shape, r2: usize, s2: Shape) -> bool {
    let r_new = max(r1, r2);

    let s1_n = broadcast_normalize(s1, r1, r_new);
    let s2_n = broadcast_normalize(s2, r2, r_new);

    let mut i = 0;
    while i < r_new {
        if s1_n[i] != 1 && s2_n[i] != 1 && s1_n[i] != s2_n[i] {
            return false;
        }
        i += 1;
    }

    true
}

const fn max(a: usize, b: usize) -> usize {
    if a >= b {
        a
    } else {
        b
    }
}

pub const fn broadcast_normalize(s: Shape, r_orig: usize, r_new: usize) -> Shape {
    assert!(r_new >= r_orig, "cannot broadcast to a lower dimension");
    assert!(
        r_orig <= shape::MAX_DIMS && r_new <= shape::MAX_DIMS,
        "cannot broadcast to a dimension higher than the max"
    );

    let r_diff = r_new - r_orig;
    let mut ret = [0; shape::MAX_DIMS];

    let mut i = 0;
    while i < r_diff {
        ret[i] = 1;
        i += 1;
    }
    while i < r_new {
        ret[i] = s[i - r_diff];
        i += 1;
    }

    ret
}

pub trait BroadcastTo<Tn: Tensor> {
    fn broadcast(self) -> Tn;
}

#[cfg(test)]
pub mod tests {
    use proptest::prelude::*;

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

        assert!(broadcast_compat(r, s, 4, [1000, 256, 256, 256, 0, 0]));
        assert!(broadcast_compat(r, s, 4, [1000, 1, 256, 256, 0, 0]));
        assert!(broadcast_compat(r, s, 2, [256, 1, 0, 0, 0, 0]));
        assert!(!broadcast_compat(r, s, 3, [1000, 256, 256, 0, 0, 0]));
    }

    proptest! {
        #[test]
        fn test_broadcast_compat_commutative(
            r1 in 0..=6,
            r2 in 0..=6,
            s1 in prop::array::uniform6(any::<usize>()),
            s2 in prop::array::uniform6(any::<usize>())
        ) {
            assert_eq!(
                broadcast_compat(r1 as usize, s1, r2 as usize, s2),
                broadcast_compat(r2 as usize, s2, r1 as usize, s1),
            );
        }
    }
}
