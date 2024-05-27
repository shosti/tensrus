pub type Shape = [usize; 6];

pub const fn stride(r: usize, s: Shape) -> Shape {
    let mut res = [0; 6];
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
    let mut out = [0; 6];
    let mut i = 0;
    while i < r {
        out[i] = s[r - i - 1];
        i += 1;
    }

    out
}
