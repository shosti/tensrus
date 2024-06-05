extern crate proc_macro;
extern crate syn;
#[macro_use]
extern crate quote;
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::ToTokens;
use syn::{
    punctuated::Punctuated, DeriveInput, Generics, Ident, Lifetime, LifetimeParam, Path,
    PathArguments, PathSegment, TraitBound, TypeParam, TypeParamBound,
};

#[proc_macro_derive(Tensor, attributes(tensor_rank, tensor_shape))]
pub fn tensor_macro_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    impl_tensor_macro(&ast)
}

fn impl_tensor_macro(ast: &DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let rank = parse_rank(ast);
    let shape = parse_shape(ast);

    let mut f_generics = ast.generics.clone();
    push_to_primitive_f_param(&mut f_generics);

    let mut generics_with_lifetime = ast.generics.clone();
    push_lifetime_param(&mut generics_with_lifetime);

    let mut generics_with_lifetime_and_rhs = generics_with_lifetime.clone();
    push_rhs_param(&mut generics_with_lifetime_and_rhs);

    let (impl_generics, type_generics, _where_clause) = ast.generics.split_for_impl();
    let (f_impl_generics, _, _) = f_generics.split_for_impl();
    let (impl_generics_with_lifetime, _, _) = generics_with_lifetime.split_for_impl();
    let (impl_generics_with_lifetime_and_rhs, _, _) =
        generics_with_lifetime_and_rhs.split_for_impl();

    let gen = quote! {
        impl #impl_generics Tensor for #name #type_generics {
            type T = T;
            type Idx = [usize; #rank];

            fn rank() -> usize {
                #rank
            }

            fn shape() -> crate::shape::Shape {
                #shape
            }

            fn num_elems() -> usize {
                crate::storage::num_elems(#rank, #shape)
            }

            fn from_fn(f: impl Fn(&Self::Idx) -> Self::T) -> Self {
                let mut v = Vec::with_capacity(Self::num_elems());
                let layout = Layout::Normal;

                for i in 0..Self::num_elems() {
                    let idx = crate::storage::nth_idx(i, #shape, layout).unwrap();
                    v.push(f(&idx));
                }

                Self {
                    storage: v.into(),
                    layout,
                }
            }

            fn map(mut self, f: impl Fn(&Self::Idx, Self::T) -> Self::T) -> Self {
                let mut next_idx = Some(Self::default_idx());
                while let Some(idx) = next_idx {
                    let i = crate::storage::storage_idx(&idx, #shape, self.layout).unwrap();
                    self.storage[i] = f(&idx, self.storage[i]);
                    next_idx = self.next_idx(&idx);
                }

                Self {
                    storage: self.storage,
                    layout: self.layout,
                }
            }

            fn set(mut self, idx: &Self::Idx, f: impl Fn(Self::T) -> Self::T) -> Self {
                let i = crate::storage::storage_idx(idx, #shape, self.layout).expect("out of bounds");
                self.storage[i] = f(self.storage[i]);

                Self {
                    storage: self.storage,
                    layout: self.layout,
                }
            }

            fn default_idx() -> Self::Idx {
                [0; #rank]
            }
            fn next_idx(&self, idx: &Self::Idx) -> Option<Self::Idx> {
                let i = crate::storage::storage_idx(idx, #shape, self.layout).ok()?;
                if i >= Self::num_elems() - 1 {
                    return None;
                }

                crate::storage::nth_idx(i + 1, #shape, self.layout).ok()
            }
        }

        impl #impl_generics crate::tensor::ShapedTensor for #name #type_generics {
            const R: usize = #rank;
            const S: Shape = #shape;
        }

        impl #impl_generics crate::tensor::BasicTensor<T> for #name #type_generics {
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }

            fn as_any_boxed(self: Box<Self>) -> Box<dyn std::any::Any> {
                self
            }

            fn clone_boxed(&self) -> Box<dyn crate::tensor::BasicTensor<T>> {
                Box::new(self.clone())
            }

            fn zeros_with_shape(&self) -> Box<dyn crate::tensor::BasicTensor<T>> {
                Box::new(Self::zeros())
            }

            fn ones_with_shape(&self) -> Box<dyn crate::tensor::BasicTensor<T>> {
                Box::new(Self::ones())
            }

            fn len(&self) -> usize {
                Self::num_elems()
            }

            // Scales `self` by `scale` and adds to `other`
            fn add(self: Box<Self>, other_basic: &dyn crate::tensor::BasicTensor<T>, scale: T) -> Box<dyn crate::tensor::BasicTensor<T>> {
                let other = Self::ref_from_basic(other_basic);
                let out = (*self * scale) + other;
                Box::new(out)
            }
        }

        impl #impl_generics std::ops::Index<&[usize; #rank]> for #name #type_generics {
            type Output = T;

            fn index(&self, idx: &[usize; #rank]) -> &Self::Output {
                let i = crate::storage::storage_idx(idx, #shape, self.layout).unwrap();
                self.storage.index(i)
            }
        }

        impl #impl_generics std::ops::Index<&[usize]> for #name #type_generics {
            type Output = T;

            fn index(&self, idx: &[usize]) -> &Self::Output {
                if idx.len() != #rank {
                    panic!("invalid index for tensor of rank {}", #rank);
                }
                let mut i = [0; #rank];
                i.copy_from_slice(idx);
                self.index(&i)
            }
        }

        impl #impl_generics crate::storage::TensorStorage<T> for #name #type_generics {
            fn storage(&self) -> &[T] {
                &self.storage
            }

            fn layout(&self) -> crate::storage::Layout {
                self.layout
            }
        }

        impl #impl_generics_with_lifetime_and_rhs crate::broadcast::BroadcastableTo<'a, T, Rhs> for #name #type_generics
            where Rhs: crate::tensor::Tensor<T = T> + crate::tensor::ShapedTensor,
            crate::type_assert::Assert<{ crate::broadcast::broadcast_compat(<Self as crate::tensor::ShapedTensor>::R, <Self as crate::tensor::ShapedTensor>::S, Rhs::R, Rhs::S) }>: crate::type_assert::IsTrue,
        {}

        impl #impl_generics_with_lifetime_and_rhs std::ops::Add<Rhs> for #name #type_generics
            where crate::view::View<'a, Self>: From<Rhs>
        {
            type Output = Self;

            fn add(self, other: Rhs) -> Self::Output {
                let view: crate::view::View<'a, Self> = other.into();
                self.map(|idx, v| v + view[idx])
            }
        }

        impl #impl_generics_with_lifetime From<crate::view::View<'a, Self>> for #name #type_generics {
            fn from(v: crate::view::View<'a, Self>) -> Self {
                Self::from_fn(|idx| v[&idx])
            }
        }

        // impl #impl_generics_with_lifetime std::ops::Add<&'a dyn crate::tensor::ShapedTensor<T, #rank, #shape>> for #name #type_generics {
        //     type Output = Self;

        //     fn add(self, other: &dyn crate::tensor::ShapedTensor<T, #rank, #shape>) -> Self {
        //         self.map(|idx, v| v + other[idx])
        //     }
        // }

        impl #impl_generics std::ops::Mul<crate::scalar::Scalar<T>> for #name #type_generics {
            type Output = Self;

            fn mul(self, other: crate::scalar::Scalar<T>) -> Self::Output {
                self * other.val()
            }
        }

        impl #impl_generics std::ops::Mul<T> for #name #type_generics {
            type Output = Self;

            fn mul(self, other: T) -> Self::Output {
                self.map(|_, v| v * other)
            }
        }


        impl #f_impl_generics std::iter::FromIterator<F> for #name #type_generics {
            fn from_iter<I>(iter: I) -> Self
            where
                I: IntoIterator<Item = F>,
            {
                let vals: Vec<T> = iter
                    .into_iter()
                    .map(|v| T::from(v).unwrap())
                    .chain(std::iter::repeat(T::zero()))
                    .take(Self::num_elems())
                    .collect();
                Self {
                    storage: vals.into(),
                    layout: crate::storage::Layout::default(),
                }
            }
        }

        impl #impl_generics std::cmp::PartialEq for #name #type_generics {
            fn eq(&self, other: &Self) -> bool {
                self.iter().all(|(idx, val)| val == other[&idx])
            }
        }

        impl #impl_generics std::cmp::Eq for #name #type_generics {}
    };
    gen.into()
}

enum RankParam {
    Usize(usize),
    Const(ConstParam),
}

impl ToTokens for RankParam {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            Self::Usize(r) => r.to_tokens(tokens),
            Self::Const(r) => r.to_tokens(tokens),
        }
    }
}

fn parse_rank(ast: &DeriveInput) -> RankParam {
    let rank_attr = ast
        .attrs
        .iter()
        .find(|&attr| attr.path().is_ident("tensor_rank"))
        .expect("tensor_rank attribute must be set");
    if let syn::Meta::NameValue(ref val) = rank_attr.meta {
        if let syn::Expr::Lit(ref lit) = val.value {
            return match lit.lit {
                syn::Lit::Int(ref int) => {
                    let val = int
                        .base10_parse()
                        .expect("unable to parse tensor_rank value");
                    RankParam::Usize(val)
                }
                syn::Lit::Str(ref s) => {
                    let val = s.value();
                    RankParam::Const(ConstParam(val))
                }
                _ => {
                    panic!("tensor_rank in unexpected format");
                }
            };
        }
    };

    panic!("tensor_rank attribute is the wrong format (should be #[tensor_rank = <integer>]");
}

struct ConstParam(String);

impl ToTokens for ConstParam {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let expr_str = format!("{{ {} }}", self.0);
        let expr: syn::Expr = syn::parse_str(expr_str.as_str()).expect("Failed to parse code");
        expr.to_tokens(tokens);
    }
}

fn parse_shape(ast: &DeriveInput) -> ConstParam {
    let shape_attr = ast
        .attrs
        .iter()
        .find(|&attr| attr.path().is_ident("tensor_shape"))
        .expect("tensor_shape attribute must be set");

    if let syn::Meta::NameValue(ref val) = shape_attr.meta {
        if let syn::Expr::Lit(ref lit) = val.value {
            if let syn::Lit::Str(ref s) = lit.lit {
                return ConstParam(s.value());
            }
        }
    };

    panic!("tensor_shape must be a string");
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

// This adds a Rhs param
fn push_rhs_param(generics: &mut Generics) {
    let ident = syn::Ident::new("Rhs", Span::call_site());
    let param = syn::GenericParam::Type(syn::TypeParam::from(ident));
    generics.params.push(param.into());
}
