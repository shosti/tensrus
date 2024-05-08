use seq_macro::seq;
use std::marker::ConstParamTy;
use std::ops::Index;

seq!(R in 0..6 {
    #[derive(ConstParamTy, PartialEq, Eq, Debug)]
    pub enum Shape {
        #(
            Rank~R([usize; R]),
        )*
    }

    impl Shape {
        pub const fn rank(self) -> usize {
            match self {
                #(
                    Self::Rank~R(_) => R,
                )*
            }
        }

        pub const fn len(self) -> usize {
            #[allow(unused_comparisons)]
            match self {
                #(
                    Self::Rank~R(dims) => {
                      let mut dim = 0;
                      let mut n = 1;

                      while dim < R {
                          n *= dims[dim];
                          dim += 1;
                      }

                      n
                    },
                )*
            }
        }

        pub const fn stride<const RANK: usize>(self) -> [usize; RANK] {
            match self {
                #(
                    Self::Rank~R(dims) => {
                        if R != RANK {
                            panic!("Shape.stride() called with invalid rank parameter");
                        }

                        let mut res = [0; RANK];
                        let mut dim = 0;
                        while dim < RANK {
                            let mut n = 1;
                            let mut d = dim + 1;
                            while d < RANK {
                                n *= dims[d];
                                d += 1;
                            }
                            res[dim] = n;

                            dim += 1;
                        }

                        res
                    }
                )*
            }
        }

        pub const fn downrank(self, n: usize) -> Shape {
            if n == 0 {
                return self
            }
            match self {
                #(
                    Self::Rank~R(dims) => {
                        seq!(R2 in 0..5 {
                            if R - 1 == R2 {
                                let mut new_dims = [0; R2];
                                let mut i = R - 1;
                                while i > 0 {
                                    new_dims[i - 1] = dims[i];
                                    i -= 1;
                                }

                                return Shape::Rank~R2(new_dims).downrank(n - 1);
                            }
                        });
                        panic!(concat!("cannot take subshape of shape with rank ", stringify!(R)));
                    },
                )*
            }
        }
    }

    impl Index<usize> for Shape {
        type Output = usize;

        fn index(&self, i: usize) -> &Self::Output {
            match self {
                #(
                    Self::Rank~R(dims) => {
                        dims.index(i)
                    },
                )*
            }
        }
    }
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_len() {
        assert_eq!(Shape::Rank0([]).len(), 1);
        assert_eq!(Shape::Rank2([2, 5]).len(), 10);
        assert_eq!(Shape::Rank3([2, 3, 3]).len(), 18);
    }

    #[test]
    fn test_stride() {
        assert_eq!(Shape::Rank0([]).stride(), []);
        assert_eq!(Shape::Rank2([2, 5]).stride(), [5, 1]);
        assert_eq!(Shape::Rank3([2, 3, 3]).stride(), [9, 3, 1]);
    }

    #[test]
    fn test_subshape() {
        assert_eq!(Shape::Rank1([12]).subshape(), Shape::Rank0([]));
        assert_eq!(Shape::Rank3([2, 3, 4]).subshape(), Shape::Rank2([3, 4]));
    }

    #[test]
    #[should_panic]
    fn test_rank0_subshape() {
        Shape::Rank0([]).subshape();
    }
}
