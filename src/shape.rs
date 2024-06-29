use crate::tensor::{Indexable, Tensor, TensorIndex};
use seq_macro::seq;

pub const MAX_DIMS: usize = 6;

pub trait Shaped {
    // Required methods
    fn rank() -> usize;
    fn shape() -> Shape;

    // Provided methods
    fn num_elems() -> usize {
        num_elems(Self::rank(), Self::shape())
    }
}

pub trait Transposable<Dest> {
    fn transpose(self) -> Dest;
}

pub trait Reducible<const DIM: usize>: Indexable {
    type Reduced: Tensor<T = Self::T, Idx = Self::Idx>;
}

pub trait Broadcastable<Dest>: Indexable
where
    Dest: Indexable<T = Self::T>,
{
    fn unbroadcasted_idx(bcast_idx: &Dest::Idx) -> Self::Idx {
        // Hopefully the implementers will ensure this, but add a runtime check
        // just in case
        debug_assert!(broadcast_compat(
            Self::rank(),
            Self::shape(),
            Dest::rank(),
            Dest::shape()
        ));
        let s_normalized = broadcast_normalize(Self::shape(), Self::rank(), Dest::rank());

        let mut src_idx = Self::Idx::default();
        let mut dim = 0;
        for i in 0..Dest::rank() {
            if s_normalized[i] == 1 && Dest::shape()[i] != 1 {
                continue;
            }
            src_idx[dim] = bcast_idx[i];
            dim += 1;
        }

        src_idx
    }
}

pub type Shape = [usize; MAX_DIMS];

pub const fn rank0() -> Shape {
    [0; MAX_DIMS]
}

seq!(R in 1..=6 {
    #(
        pub const fn rank~R(dims: [usize; R]) -> Shape {
            let mut out = [0; MAX_DIMS];
            let mut i = 0;
            while i < R {
                out[i] = dims[i];
                i += 1;
            }

            out
        }
    )*
});

pub const fn num_elems(r: usize, s: Shape) -> usize {
    let mut dim = 0;
    let mut n = 1;

    while dim < r {
        n *= s[dim];
        dim += 1;
    }

    n
}

pub const fn stride(r: usize, s: Shape) -> [usize; MAX_DIMS] {
    let mut res = [0; MAX_DIMS];
    let mut dim = 0;

    while dim < r {
        let mut i = dim + 1;
        let mut cur = 1;
        while i < r {
            cur *= s[i];
            i += 1;
        }
        res[dim] = cur;

        dim += 1;
    }

    res
}

pub const fn transpose(r: usize, s: Shape) -> Shape {
    let mut out = [0; MAX_DIMS];
    let mut i = 0;
    while i < r {
        out[i] = s[r - i - 1];
        i += 1;
    }

    out
}

// Returns the tensor shape when reducing along dimension `dim`
pub const fn reduced_shape(r: usize, s: Shape, dim: usize) -> Shape {
    assert!(
        dim < r,
        "cannot reduce along a dimension greater than the original rank"
    );
    let mut out = [0; MAX_DIMS];
    let mut i = 0;
    while i < r {
        if i == dim {
            out[i] = 1;
        } else {
            out[i] = s[i];
        }
        i += 1;
    }

    out
}

pub const fn shapes_equal(r1: usize, s1: Shape, r2: usize, s2: Shape) -> bool {
    if r1 != r2 {
        return false;
    }
    let mut i = 0;
    while i < r1 {
        if s1[i] != s2[i] {
            return false;
        }
        i += 1;
    }
    true
}

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
        r_src <= MAX_DIMS && r_dest <= MAX_DIMS,
        "cannot broadcast to a dimension higher than the max"
    );

    let r_diff = r_dest - r_src;
    let mut ret = [0; MAX_DIMS];

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

pub const fn subtensor_shape(n: usize, r: usize, s: Shape) -> Shape {
    assert!(
        n <= r,
        "cannot take a subtensor with more dimensions than the original rank"
    );

    let mut out = [0; MAX_DIMS];
    let mut i = 0;
    while i < r - n {
        out[i] = s[i + n];
        i += 1;
    }

    out
}

#[cfg(test)]
pub mod tests {

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
    fn test_reduced_shape() {
        assert_eq!(reduced_shape(3, [2, 3, 4, 0, 0, 0], 0), [1, 3, 4, 0, 0, 0]);
        assert_eq!(reduced_shape(3, [2, 3, 4, 0, 0, 0], 1), [2, 1, 4, 0, 0, 0]);
        assert_eq!(reduced_shape(3, [2, 3, 4, 0, 0, 0], 2), [2, 3, 1, 0, 0, 0]);
    }

    #[test]
    fn test_subtensor_shape() {
        let s = rank6([1, 2, 3, 4, 5, 6]);
        let r = 6;
        assert_eq!(subtensor_shape(0, r, s), s);
        assert_eq!(subtensor_shape(1, r, s), rank5([2, 3, 4, 5, 6]));
        assert_eq!(subtensor_shape(2, r, s), rank4([3, 4, 5, 6]));
        assert_eq!(subtensor_shape(3, r, s), rank3([4, 5, 6]));
        assert_eq!(subtensor_shape(4, r, s), rank2([5, 6]));
        assert_eq!(subtensor_shape(5, r, s), rank1([6]));
        assert_eq!(subtensor_shape(6, r, s), rank0());
    }
}
