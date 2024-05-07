use seq_macro::seq;
use std::marker::ConstParamTy;
use std::ops::Index;

seq!(R in 0..6 {
    #[derive(ConstParamTy, PartialEq, Eq)]
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
}
