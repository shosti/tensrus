extern crate proc_macro;
extern crate syn;
#[macro_use]
extern crate quote;
use proc_macro::TokenStream;
use proc_macro2::Span;
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

            fn add(self: Box<Self>, other: &Box<dyn crate::tensor::BasicTensor<T>>) -> Box<dyn crate::tensor::BasicTensor<T>> {
                Box::new(crate::tensor::Tensor::map(*self, |idx, val| val + other[idx]))
            }

            fn zeros_with_shape(&self) -> Box<dyn crate::tensor::BasicTensor<T>> {
                Box::new(<Self as crate::tensor::Tensor>::zeros())
            }

            fn ones_with_shape(&self) -> Box<dyn crate::tensor::BasicTensor<T>> {
                Box::new(<Self as crate::tensor::Tensor>::ones())
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
