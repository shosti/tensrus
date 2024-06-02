pub const MAX_DIMS: usize = 6;
pub type Shape = [usize; MAX_DIMS];

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
