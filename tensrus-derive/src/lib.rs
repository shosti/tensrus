extern crate proc_macro;
extern crate syn;
#[macro_use]
extern crate quote;
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::ToTokens;
use syn::{
    punctuated::Punctuated, AngleBracketedGenericArguments, DeriveInput, Generics, Ident, Lifetime,
    LifetimeParam, Path, PathArguments, PathSegment, TraitBound, TypeParam, TypeParamBound,
};

#[proc_macro_derive(Tensor)]
pub fn tensor_macro_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    impl_tensor_macro(&ast)
}

fn impl_tensor_macro(ast: &DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let (wrapped_type, wrapped_type_args) = get_wrapped_type(ast);
    let rank = get_rank(&wrapped_type_args);

    let mut f_generics = ast.generics.clone();
    push_to_primitive_f_param(&mut f_generics);

    let mut generics_with_lifetime = ast.generics.clone();
    push_lifetime_param(&mut generics_with_lifetime);

    let (impl_generics, type_generics, where_clause) = ast.generics.split_for_impl();
    let (f_impl_generics, _, _) = f_generics.split_for_impl();
    let (impl_generics_with_lifetime, _, _) = generics_with_lifetime.split_for_impl();
    let gen = quote! {
        impl #impl_generics crate::tensor::Tensor for #name #type_generics #where_clause {
            type T = T;
            type Idx = [usize; #rank];

            fn set(self, idx: &Self::Idx, val: Self::T) -> Self {
                Self(self.0.set(idx, val))
            }

            fn map(self, f: impl Fn(&Self::Idx, Self::T) -> Self::T) -> Self {
                Self(self.0.map(f))
            }

            fn default_idx() -> Self::Idx {
                #wrapped_type::#wrapped_type_args::default_idx()
            }

            fn next_idx(&self, idx: &Self::Idx) -> Option<Self::Idx> {
                self.0.next_idx(idx)
            }

            fn repeat(n: Self::T) -> Self {
                Self(#wrapped_type::repeat(n))
            }

            fn num_elems() -> usize {
                #wrapped_type::#wrapped_type_args::num_elems()
            }
        }

        impl #impl_generics crate::tensor::BasicTensor<T> for #name #type_generics #where_clause {
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }

            fn as_any_boxed(self: Box<Self>) -> Box<dyn std::any::Any> {
                self
            }

            fn num_elems(&self) -> usize {
                self.0.num_elems()
            }

            fn clone_boxed(&self) -> Box<dyn crate::tensor::BasicTensor<T>> {
                Box::new(self.clone())
            }

            fn zeros_with_shape(&self) -> Box<dyn crate::tensor::BasicTensor<T>> {
                Box::new(<Self as crate::tensor::Tensor>::zeros())
            }

            fn ones_with_shape(&self) -> Box<dyn crate::tensor::BasicTensor<T>> {
                Box::new(<Self as crate::tensor::Tensor>::ones())
            }

            fn add(self: Box<Self>, other_basic: &dyn crate::tensor::BasicTensor<T>, scale: T) -> Box<dyn crate::tensor::BasicTensor<T>> {
                let other = <Self as crate::tensor::Tensor>::ref_from_basic(other_basic);
                let out = (*self * scale) + other;
                Box::new(out)
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
            type Item = ([usize; #rank], T);
            type IntoIter = crate::tensor::TensorIterator<'a, #name #type_generics>;

            fn into_iter(self) -> Self::IntoIter {
                Self::IntoIter::new(self)
            }
        }

        impl #impl_generics_with_lifetime std::ops::Add<&'a Self> for #name #type_generics #where_clause {
            type Output = Self;

            fn add(self, other: &Self) -> Self::Output {
                Self(self.0 + &other.0)
            }
        }

        impl #impl_generics std::ops::Mul<T> for #name #type_generics #where_clause {
            type Output = Self;

            fn mul(self, other: T) -> Self::Output {
                Self(self.0 * other)
            }
        }

        impl #impl_generics std::ops::Index<&[usize; #rank]> for #name #type_generics #where_clause {
            type Output = T;

            fn index(&self, idx: &[usize; #rank]) -> &Self::Output {
                self.0.index(idx)
            }
        }

        impl #impl_generics std::ops::Index<&[usize]> for #name #type_generics #where_clause {
            type Output = T;

            fn index(&self, idx: &[usize]) -> &Self::Output {
                self.0.index(idx)
            }
        }

        impl #impl_generics std::ops::Mul<crate::scalar::Scalar<T>> for #name #type_generics #where_clause {
            type Output = Self;

            fn mul(self, other: crate::scalar::Scalar<T>) -> Self::Output {
                Self(self.0 * other)
            }
        }

        impl #impl_generics Clone for #name #type_generics #where_clause {
            fn clone(&self) -> Self {
                Self(self.0.clone())
            }
        }

        impl #impl_generics From<#wrapped_type #wrapped_type_args> for #name #type_generics #where_clause {
            fn from(t: #wrapped_type #wrapped_type_args) -> Self {
                Self(t)
            }
        }

        impl #impl_generics From<#name #type_generics> for #wrapped_type #wrapped_type_args #where_clause {
            fn from(s: #name #type_generics) -> Self {
                s.0
            }
        }
    };
    gen.into()
}

#[proc_macro_derive(Tensor2, attributes(tensor_rank, tensor_shape))]
pub fn tensor2_macro_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    impl_tensor2_macro(&ast)
}

fn impl_tensor2_macro(ast: &DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let rank = parse_rank(ast);
    let shape = parse_shape(ast);

    let mut f_generics = ast.generics.clone();
    push_to_primitive_f_param(&mut f_generics);

    let mut generics_with_lifetime = ast.generics.clone();
    push_lifetime_param(&mut generics_with_lifetime);

    let (impl_generics, type_generics, _where_clause) = ast.generics.split_for_impl();
    let (f_impl_generics, _, _) = f_generics.split_for_impl();
    let (impl_generics_with_lifetime, _, _) = generics_with_lifetime.split_for_impl();

    let gen = quote! {
        impl #impl_generics Tensor2 for #name #type_generics {
            type T = T;
            type Idx = [usize; #rank];

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

        impl #impl_generics crate::tensor2::BasicTensor<T> for #name #type_generics {
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }

            fn as_any_boxed(self: Box<Self>) -> Box<dyn std::any::Any> {
                self
            }

            fn clone_boxed(&self) -> Box<dyn crate::tensor2::BasicTensor<T>> {
                Box::new(self.clone())
            }

            fn zeros_with_shape(&self) -> Box<dyn crate::tensor2::BasicTensor<T>> {
                Box::new(Self::zeros())
            }

            fn ones_with_shape(&self) -> Box<dyn crate::tensor2::BasicTensor<T>> {
                Box::new(Self::ones())
            }

            fn len(&self) -> usize {
                Self::num_elems()
            }

            // Scales `self` by `scale` and adds to `other`
            fn add(self: Box<Self>, other_basic: &dyn crate::tensor2::BasicTensor<T>, scale: T) -> Box<dyn crate::tensor2::BasicTensor<T>> {
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

        impl #impl_generics_with_lifetime std::ops::Add<&'a Self> for #name #type_generics {
            type Output = Self;

            fn add(self, other: &Self) -> Self::Output {
                self.map(|idx, v| v + other[idx])
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

fn get_wrapped_type(ast: &DeriveInput) -> (Ident, AngleBracketedGenericArguments) {
    let data: &syn::DataStruct;
    if let syn::Data::Struct(d) = &ast.data {
        data = d;
    } else {
        panic!("Tensor can only be derived for struct types");
    }
    let field: &syn::Field;
    if let syn::Fields::Unnamed(ref f) = data.fields {
        field = f
            .unnamed
            .first()
            .expect("Wrapped tensor type should be the first wrapped field");
    } else {
        panic!("Tensor can only be derived for wrapper structs");
    }
    let type_path: &syn::TypePath;
    if let syn::Type::Path(ref p) = field.ty {
        type_path = p;
    } else {
        panic!("Invalid wrapped type when deriving Tensor");
    }

    let last_segment = type_path.path.segments.last().unwrap();
    let args: &AngleBracketedGenericArguments;
    if let syn::PathArguments::AngleBracketed(ref a) = last_segment.arguments {
        args = a;
    } else {
        panic!("Expected wrapped Tensor type to have generic arguments");
    }

    (last_segment.ident.clone(), args.clone())
}

fn get_rank(args: &AngleBracketedGenericArguments) -> syn::LitInt {
    let rank_arg = &args.args[1];
    let rank_expr: &syn::ExprLit;
    if let syn::GenericArgument::Const(syn::Expr::Lit(ref e)) = rank_arg {
        rank_expr = e;
    } else {
        panic!("Deriving Tensor: expected wrapped type's second generic argument to be a constant specifying the tensor's rank");
    }
    let lit: &syn::LitInt;
    if let syn::Lit::Int(ref l) = rank_expr.lit {
        lit = l;
    } else {
        panic!("Deriving Tensor: expected rank to be an integer");
    }

    lit.clone()
}
