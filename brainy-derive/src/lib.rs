extern crate proc_macro;
extern crate syn;
#[macro_use]
extern crate quote;
use proc_macro::TokenStream;

#[proc_macro_derive(Tensor)]
pub fn tensor_macro_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    impl_tensor_macro(&ast)
}

fn impl_tensor_macro(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let gen = quote! {
        impl<T: Numeric, const R: usize, const S: TensorShape> Tensor<T, R, S> for #name<T, R, S> {
            fn shape(&self) -> [usize; R] {
                self.0.shape()
            }

            fn get(&self, idx: &[usize; R]) -> Result<T, IndexError> {
                self.0.get(idx)
            }
        }

        impl<T: Numeric, const R: usize, const S: TensorShape, F> FromIterator<F> for #name<T, R, S>
        where
            F: ToPrimitive,
        {
            fn from_iter<I>(iter: I) -> Self
            where
                I: IntoIterator<Item = F>,
            {
                Self(iter.into_iter().collect())
            }
        }
    };
    gen.into()
}
