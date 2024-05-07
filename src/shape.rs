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
            match self {
                #(
                    Self::Rank~R(dims) => {
                        Dims { dims }.len()
                    },
                )*
            }
        }
    }

    #(
        impl Into<Dims<R>> for Shape {
            fn into(self) -> Dims<R> {
                if let Self::Rank~R(dims) = self {
                    Dims { dims }
                } else {
                    panic!("converting to invalid dimension");
                }
            }
        }
    )*
});
