use crate::{
    shape::{self, Shape},
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

#[cfg(test)]
pub mod tests {
    use crate::{
        generic_tensor::GenericTensor, matrix::Matrix, scalar::Scalar, shape::MAX_DIMS,
        vector::Vector,
    };

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
        let m: Matrix<_, 3, 3> = v.view().to_generic().broadcast().from_generic().into();

        assert_eq!(
            m,
            Matrix::<f64, _, _>::from([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        );

        let t: GenericTensor<_, 3, { [3; MAX_DIMS] }> = v.view().to_generic().broadcast().into();
        assert_eq!(
            t,
            [1, 2, 3]
                .into_iter()
                .cycle()
                .collect::<GenericTensor<f64, 3, { [3; MAX_DIMS] }>>()
        );

        let x: Matrix<f64, _, _> = Matrix::from([[1, 2, 3], [4, 5, 6]]);
        let one = Scalar::<f64>::from(1);
        let one_gen = one.view().to_generic();
        let ones: View<Matrix<_, 2, 3>> = one_gen.broadcast().from_generic();

        assert_eq!(x + ones, Matrix::<f64, _, _>::from([[2, 3, 4], [5, 6, 7]]));
    }
}
