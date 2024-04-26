extern crate proc_macro;
extern crate syn;
#[macro_use]
extern crate quote;
use proc_macro::TokenStream;
use syn::{Meta, Expr, Lit, Attribute, DeriveInput};

#[proc_macro_derive(Tensor, attributes(TensorRank))]
pub fn tensor_macro_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    impl_tensor_macro(&ast)
}

fn parse_rank(rank_attr: &Attribute) -> usize {
    if let Meta::NameValue(ref val) = rank_attr.meta {
        if let Expr::Lit(ref lit) = val.value {
            if let Lit::Int(ref int) = lit.lit {
                return int.base10_parse().expect("unable to parse TensorRank value");
            }
        }
    };

    panic!("TensorRank attribute is the wrong format (should be #[TensorRank = <int>]");
}

fn impl_tensor_macro(ast: &DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let (impl_generics, type_generics, where_clause) = ast.generics.split_for_impl();
    let rank_attr = &ast
        .attrs
        .iter()
        .find(|&attr| attr.path().is_ident("TensorRank"))
        .expect("TensorRank attribute must be set");
    let rank = parse_rank(rank_attr);
    let gen = quote! {
        impl #impl_generics Tensor<T, #rank> for #name #type_generics #where_clause {
            // fn from_fn<F>(cb: F) -> Self
            // where
            //     F: FnMut([usize; R]) -> T,
            // {
            //     let t:
            //     Self(GenericTensor::from_fn(cb))
            // }

            fn shape(&self) -> [usize; #rank] {
                self.0.shape()
            }

            fn get(&self, idx: &[usize; #rank]) -> Result<T, IndexError> {
                self.0.get(idx)
            }

            fn get_at_idx(&self, i: usize) -> Result<T, IndexError> {
                self.0.get_at_idx(i)
            }

            fn update(&self, f: &dyn Fn(T) -> T) {
                self.0.update(f);
            }
        }

        // impl<T: Numeric, const R: usize, const S: TensorShape, F> FromIterator<F> for #name<T, R, S>
        // where
        //     F: ToPrimitive,
        // {
        //     fn from_iter<I>(iter: I) -> Self
        //     where
        //         I: IntoIterator<Item = F>,
        //     {
        //         Self(iter.into_iter().collect())
        //     }
        // }

        // impl<'a, T: Numeric, const R: usize, const S: TensorShape> IntoIterator for &'a #name<T, R, S> {
        //     type Item = T;
        //     type IntoIter = TensorIterator<'a, T, R, S>;

        //     fn into_iter(self) -> Self::IntoIter {
        //         Self::IntoIter::new(self)
        //     }
        // }

        // impl<T: Numeric, const R: usize, const S: TensorShape> std::ops::Add<T> for #name<T, R, S> {
        //     type Output = Self;

        //     fn add(self, other: T) -> Self::Output {
        //         Self(self.0 + other)
        //     }
        // }

        // impl<T: Numeric, const R: usize, const S: TensorShape> std::ops::Add<crate::scalar::Scalar<T>> for #name<T, R, S> {
        //     type Output = Self;

        //     fn add(self, other: crate::scalar::Scalar<T>) -> Self::Output {
        //         Self(self.0 + other)
        //     }
        // }

        // impl<T: Numeric, const R: usize, const S: TensorShape> std::ops::AddAssign<T> for #name<T, R, S> {
        //     fn add_assign(&mut self, other: T) {
        //         self.0 += other;
        //     }
        // }

        // impl<T: Numeric, const R: usize, const S: TensorShape> std::ops::Mul<T> for #name<T, R, S> {
        //     type Output = Self;

        //     fn mul(self, other: T) -> Self::Output {
        //         Self(self.0 * other)
        //     }
        // }

        // impl<T: Numeric, const R: usize, const S: TensorShape> std::ops::Mul<crate::scalar::Scalar<T>> for #name<T, R, S> {
        //     type Output = Self;

        //     fn mul(self, other: crate::scalar::Scalar<T>) -> Self::Output {
        //         Self(self.0 * other)
        //     }
        // }

        // impl<T: Numeric, const R: usize, const S: TensorShape> std::ops::MulAssign<T> for #name<T, R, S> {
        //     fn mul_assign(&mut self, other: T) {
        //         self.0 *= other;
        //     }
        // }

        // impl<T: Numeric, const R: usize, const S: TensorShape> Clone for #name<T, R, S> {
        //     fn clone(&self) -> Self {
        //         Self(self.0.clone())
        //     }
        // }

        // impl<T: Numeric, const R: usize, const S: TensorShape> crate::tensor::TensorOps<T> for #name<T, R, S> {}
    };
    gen.into()
}
