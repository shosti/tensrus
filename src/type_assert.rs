use std::marker::PhantomData;

pub struct Assert<const T: bool> {
    _prevent_contstruction: PhantomData<()>,
}

impl const IsTrue for Assert<true> {}

#[const_trait]
pub trait IsTrue {}
