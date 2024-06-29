extern crate proc_macro;
extern crate syn;
#[macro_use]
extern crate quote;
use proc_macro::TokenStream;

#[proc_macro_derive(TensorStorage)]
pub fn tensor_storage_macro_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    impl_tensor_storage_macro(&ast)
}

#[proc_macro_derive(OwnedTensorStorage)]
pub fn owned_tensor_storage_macro_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    impl_owned_tensor_storage_macro(&ast)
}

#[proc_macro_derive(Tensor)]
pub fn tensor_macro_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    impl_tensor_macro(&ast)
}

fn impl_tensor_storage_macro(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let (impl_generics, type_generics, _where_clause) = ast.generics.split_for_impl();

    let gen = quote! {
        impl #impl_generics crate::storage::TensorStorage<T> for #name #type_generics {
            fn storage(&self) -> &crate::storage::Storage<T> {
                &self.storage
            }
        }

        impl #impl_generics crate::storage::TensorLayout for #name #type_generics {
            fn layout(&self) -> crate::storage::Layout {
                self.storage.layout
            }
        }
    };

    gen.into()
}

fn impl_owned_tensor_storage_macro(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let (impl_generics, type_generics, _where_clause) = ast.generics.split_for_impl();

    let gen = quote! {
        impl #impl_generics crate::storage::OwnedTensorStorage<T> for #name #type_generics {
            fn into_storage(self) -> crate::storage::Storage<T> {
                self.storage
            }
            fn from_storage(storage: crate::storage::Storage<T>) -> Self {
                Self { storage }
            }
        }
    };

    gen.into()
}

fn impl_tensor_macro(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let (impl_generics, type_generics, _where_clause) = ast.generics.split_for_impl();
    let mut generics_with_rhs = ast.generics.clone();
    push_rhs_param(&mut generics_with_rhs);
    let (impl_generics_with_rhs, _, _) = generics_with_rhs.split_for_impl();

    let gen = quote! {
        impl #impl_generics Tensor for #name #type_generics {
        }

        impl #impl_generics Default for #name #type_generics
        where
            T: Default + Copy,
        {
            fn default() -> Self {
                Self::repeat(T::default())
            }
        }

        impl #impl_generics std::iter::FromIterator<T> for #name #type_generics
        where
            T: Default
        {
            fn from_iter<I>(iter: I) -> Self
            where
                I: IntoIterator<Item = T>,
            {
                let storage = crate::storage::Storage::<T>::from_iter(iter, Self::num_elems());
                <Self as crate::storage::OwnedTensorStorage<T>>::from_storage(storage)
            }
        }

        impl #impl_generics std::iter::IntoIterator for #name #type_generics {
            type Item = (<Self as crate::tensor::Indexable>::Idx, <Self as crate::tensor::Indexable>::T);
            type IntoIter = crate::iterator::IntoIter<Self>;

            fn into_iter(self) -> Self::IntoIter {
                crate::iterator::IntoIter::new(self)
            }
        }

        impl #impl_generics std::cmp::PartialEq for #name #type_generics where T: PartialEq {
            fn eq(&self, other: &Self) -> bool {
                self.iter().all(|(idx, val)| val == other.index(&idx))
            }
        }

        impl #impl_generics std::cmp::PartialEq<&Self> for #name #type_generics where T: PartialEq {
            fn eq(&self, other: &&Self) -> bool {
                self.eq(*other)
            }
        }


        impl #impl_generics std::cmp::Eq for #name #type_generics where T: Eq {
        }

        impl #impl_generics_with_rhs std::ops::Add<Rhs> for #name #type_generics
        where
            Rhs: crate::tensor::Indexable<Idx = <Self as crate::tensor::Indexable>::Idx, T = <Self as crate::tensor::Indexable>::T>,
            T: std::ops::Add<T, Output = T> + Clone,
        {
            type Output = Self;

            fn add(self, rhs: Rhs) -> Self::Output {
                self.map(|idx, val| val.clone() + rhs[idx].clone())
            }
        }

        impl #impl_generics std::ops::Mul<T> for #name #type_generics
        where
            T: std::ops::Mul<T, Output = T> + Clone,
        {
            type Output = Self;

            fn mul(self, rhs: T) -> Self::Output {
                self.map(|idx, val| val.clone() * rhs.clone())
            }
        }

        impl #impl_generics std::ops::Mul<crate::scalar::Scalar<T>> for #name #type_generics
        where
            T: std::ops::Mul<T, Output = T> + Clone,
        {
            type Output = Self;

            fn mul(self, rhs: crate::scalar::Scalar<T>) -> Self::Output {
                self * rhs.val()
            }
        }
    };

    gen.into()
}

// This adds a Rhs param
fn push_rhs_param(generics: &mut syn::Generics) {
    let ident = syn::Ident::new("Rhs", proc_macro2::Span::call_site());
    let param = syn::GenericParam::Type(syn::TypeParam::from(ident));
    generics.params.push(param.into());
}
