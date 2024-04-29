extern crate proc_macro;
extern crate syn;
#[macro_use]
extern crate quote;
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::ToTokens;
use syn::{
    punctuated::Punctuated, DeriveInput, Expr, Generics, Ident, Lifetime, LifetimeParam, Lit, Meta,
    Path, PathArguments, PathSegment, TraitBound, TypeParam, TypeParamBound,
};

#[proc_macro_derive(Tensor, attributes(tensor_rank, tensor_shape))]
pub fn tensor_macro_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    impl_tensor_macro(&ast)
}

fn impl_tensor_macro(ast: &DeriveInput) -> TokenStream {
    let name = &ast.ident;

    let mut f_generics = ast.generics.clone();
    push_to_primitive_f_param(&mut f_generics);

    let mut generics_with_lifetime = ast.generics.clone();
    push_lifetime_param(&mut generics_with_lifetime);

    let (impl_generics, type_generics, where_clause) = ast.generics.split_for_impl();
    let (f_impl_generics, _, _) = f_generics.split_for_impl();
    let (impl_generics_with_lifetime, _, _) = generics_with_lifetime.split_for_impl();
    let rank = parse_rank(&ast);
    let shape = parse_shape(&ast);
    let gen = quote! {
        impl #impl_generics Tensor for #name #type_generics #where_clause {}

        impl #impl_generics crate::tensor::ShapedTensor<T, #rank, #shape> for #name #type_generics #where_clause {
            fn get(&self, idx: [usize; #rank]) -> T {
                self.0.get(idx)
            }
            fn set(&self, idx: [usize; #rank], val: T) {
                self.0.set(idx, val)
            }
        }

        impl #f_impl_generics FromIterator<F> for #name #type_generics #where_clause
        {
            fn from_iter<I>(iter: I) -> Self
            where
                I: IntoIterator<Item = F>,
            {
                Self(iter.into_iter().collect())
            }
        }

        impl #impl_generics_with_lifetime IntoIterator for &'a #name #type_generics #where_clause {
            type Item = T;
            type IntoIter = crate::tensor::TensorIterator<'a, T, #rank, #shape, #name #type_generics>;

            fn into_iter(self) -> Self::IntoIter {
                Self::IntoIter::new(self)
            }
        }

        impl #impl_generics std::ops::Add<T> for #name #type_generics #where_clause {
            type Output = Self;

            fn add(self, other: T) -> Self::Output {
                Self(self.0 + other)
            }
        }

        impl #impl_generics std::ops::Add<crate::scalar::Scalar<T>> for #name #type_generics #where_clause {
            type Output = Self;

            fn add(self, other: crate::scalar::Scalar<T>) -> Self::Output {
                Self(self.0 + other)
            }
        }

        impl #impl_generics std::ops::AddAssign<T> for #name #type_generics #where_clause {
            fn add_assign(&mut self, other: T) {
                self.0 += other;
            }
        }

        impl #impl_generics std::ops::Mul<T> for #name #type_generics #where_clause {
            type Output = Self;

            fn mul(self, other: T) -> Self::Output {
                Self(self.0 * other)
            }
        }

        impl #impl_generics std::ops::Mul<crate::scalar::Scalar<T>> for #name #type_generics #where_clause {
            type Output = Self;

            fn mul(self, other: crate::scalar::Scalar<T>) -> Self::Output {
                Self(self.0 * other)
            }
        }

        impl #impl_generics std::ops::MulAssign<T> for #name #type_generics #where_clause {
            fn mul_assign(&mut self, other: T) {
                self.0 *= other;
            }
        }

        impl #impl_generics Clone for #name #type_generics #where_clause {
            fn clone(&self) -> Self {
                Self(self.0.clone())
            }
        }

        impl #impl_generics crate::tensor::TensorOps<T> for #name #type_generics #where_clause {
            fn zeros() -> Self {
                Self(crate::generic_tensor::GenericTensor::zeros())
            }

            fn update<F: Fn(T) -> T>(&mut self, f: F) {
                self.0.update(f);
            }

            fn update_zip<F: Fn(T, T) -> T>(&mut self, other: &Self, f: F) {
                self.0.update_zip(&other.0, f);
            }
        }
    };
    gen.into()
}

// This whole hullabaloo is to add <F: ToPrimitive> to the generics
// clause. Thanks ChatGPT!
fn push_to_primitive_f_param(generics: &mut Generics) {
    let ident = Ident::new("F", Span::call_site());

    let mut segments = Punctuated::new();
    segments.push(PathSegment {
        ident: Ident::new("num", Span::call_site()),
        arguments: PathArguments::None,
    });
    segments.push(PathSegment {
        ident: Ident::new("ToPrimitive", Span::call_site()),
        arguments: PathArguments::None,
    });
    let path = Path {
        leading_colon: None,
        segments,
    };

    let trait_bound = TraitBound {
        paren_token: None,
        modifier: syn::TraitBoundModifier::None,
        lifetimes: None,
        path,
    };

    let bound = TypeParamBound::Trait(trait_bound);

    let mut bounds = Punctuated::new();
    bounds.push(bound);

    let type_param = TypeParam {
        attrs: vec![],
        ident,
        colon_token: Some(Default::default()),
        bounds,
        default: None,
        eq_token: None,
    };

    generics.params.push(type_param.into());
}

// This adds the 'a lifetime to a list of params
fn push_lifetime_param(generics: &mut Generics) {
    let param = LifetimeParam::new(Lifetime::new("'a", Span::call_site()));
    generics.params.insert(0, param.into());
}

fn parse_rank(ast: &DeriveInput) -> usize {
    let rank_attr = ast
        .attrs
        .iter()
        .find(|&attr| attr.path().is_ident("tensor_rank"))
        .expect("tensor_rank attribute must be set");
    if let Meta::NameValue(ref val) = rank_attr.meta {
        if let Expr::Lit(ref lit) = val.value {
            if let Lit::Int(ref int) = lit.lit {
                return int
                    .base10_parse()
                    .expect("unable to parse tensor_rank value");
            }
        }
    };

    panic!("tensor_rank attribute is the wrong format (should be #[tensor_rank = <integer>]");
}

fn parse_shape(ast: &DeriveInput) -> ShapeParam {
    let shape_attr = ast
        .attrs
        .iter()
        .find(|&attr| attr.path().is_ident("tensor_shape"))
        .expect("tensor_shape attribute must be set");

    if let Meta::NameValue(ref val) = shape_attr.meta {
        if let Expr::Lit(ref lit) = val.value {
            if let Lit::Str(ref s) = lit.lit {
                return ShapeParam(s.value());
            }
        }
    };

    panic!("tensor_shape must be a string");
}

struct ShapeParam(String);

impl ToTokens for ShapeParam {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let expr_str = format!("{{ {} }}", self.0);
        let expr: syn::Expr = syn::parse_str(expr_str.as_str()).expect("Failed to parse code");
        expr.to_tokens(tokens);
    }
}
