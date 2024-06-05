pub const MAX_DIMS: usize = 6;
pub type Shape = [usize; MAX_DIMS];

pub trait Shaped {
    const R: usize;
    const S: Shape;
}

pub const fn stride(r: usize, s: Shape) -> Shape {
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

pub const fn transpose_shape(r: usize, s: Shape) -> Shape {
    let mut out = [0; MAX_DIMS];
    let mut i = 0;
    while i < r {
        out[i] = s[r - i - 1];
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

// Returns the tensor shape when downranking by 1
pub const fn subtensor_shape(r: usize, s: Shape) -> Shape {
    if r == 0 {
        panic!("cannot take subtensor of tensor of rank 0");
    }
    let mut out = [0; MAX_DIMS];
    let mut i = r - 1;
    while i > 0 {
        out[i - 1] = s[i];
        i -= 1;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduced_shape() {
        assert_eq!(reduced_shape(3, [2, 3, 4, 0, 0, 0], 0), [1, 3, 4, 0, 0, 0]);
        assert_eq!(reduced_shape(3, [2, 3, 4, 0, 0, 0], 1), [2, 1, 4, 0, 0, 0]);
        assert_eq!(reduced_shape(3, [2, 3, 4, 0, 0, 0], 2), [2, 3, 1, 0, 0, 0]);
    }
}
