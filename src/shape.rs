use seq_macro::seq;
use std::marker::ConstParamTy;

#[derive(ConstParamTy, PartialEq, Eq)]
pub struct Dims<const R: usize> {
    pub dims: [usize; R],
}

impl<const R: usize> Dims<R> {
    pub const fn len(self) -> usize {
        let mut dim = 0;
        let mut n = 1;

        while dim < R {
            n *= self.dims[dim];
            dim += 1;
        }

        n
    }
}

seq!(R in 0..6 {
    #[derive(ConstParamTy, PartialEq, Eq)]
    pub enum Shape {
        #(
            Rank~R(Dims<R>),
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
            match self {
                #(
                    Self::Rank~R(dims) => {
                        dims.len()
                    },
                )*
            }
        }

        pub const fn dims<const RANK: usize>(self) -> Dims<RANK> {
            match self {
                #(
                    Self::Rank~R(dims) => {
                        if RANK == R {
                            return dims;
                        }

                        panic!("no dimensions for given rank");
                    },
                )*
            }
        }
    }
});
