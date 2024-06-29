use crate::{
    storage::{Layout, Storage},
    tensor::{Indexable, Tensor, TensorIndex},
};
use std::{iter::Map, marker::PhantomData};

pub struct Iter<'a, Tn>
where
    Tn: Indexable,
{
    t: &'a Tn,
    cur: Option<Tn::Idx>,
}

impl<'a, Tn> Iter<'a, Tn>
where
    Tn: Indexable,
{
    pub fn new(t: &'a Tn) -> Self {
        Self {
            t,
            cur: Some(Tn::Idx::default()),
        }
    }

    pub fn values(self) -> Map<Iter<'a, Tn>, impl FnMut((Tn::Idx, &'a Tn::T)) -> &'a Tn::T> {
        self.map(|(_, v)| v)
    }
}

impl<'a, Tn> Iterator for Iter<'a, Tn>
where
    Tn: Indexable,
{
    type Item = (Tn::Idx, &'a Tn::T);

    fn next(&mut self) -> Option<Self::Item> {
        match &self.cur {
            None => None,
            Some(idx) => {
                let cur_idx = *idx;
                let item = (cur_idx, self.t.index(&cur_idx));
                self.cur = self.t.next_idx(&cur_idx);

                Some(item)
            }
        }
    }
}

pub struct IntoIter<Tn>
where
    Tn: Tensor,
{
    cur: usize,
    iter: std::vec::IntoIter<Tn::T>,
    layout: Layout,
    _marker: PhantomData<Tn>,
}

impl<Tn> IntoIter<Tn>
where
    Tn: Tensor,
{
    pub fn new(t: Tn) -> Self {
        let storage = t.into_storage();
        let layout = storage.layout;
        let vec: Vec<Tn::T> = storage.data.into();
        let iter = vec.into_iter();

        Self {
            cur: 0,
            iter,
            layout,
            _marker: PhantomData,
        }
    }
}

impl<Tn> Iterator for IntoIter<Tn>
where
    Tn: Tensor,
{
    type Item = (Tn::Idx, Tn::T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur >= Tn::num_elems() {
            debug_assert!(
                self.iter.next().is_none(),
                "storage had more elements than specified by tensor shape"
            );
            return None;
        }

        let mut idx = Tn::Idx::default();
        Storage::<Tn::T>::get_nth_idx(self.cur, idx.as_mut(), Tn::rank(), Tn::shape(), self.layout)
            .unwrap();
        self.cur += 1;
        let val = self
            .iter
            .next()
            .expect("storage had fewer elements than specified by tensor shape");

        Some((idx, val))
    }
}

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;

    #[test]
    fn test_into_iter() {
        let mut iter = Matrix::from([[1, 2], [3, 4]]).into_iter();

        assert_eq!(iter.next().unwrap(), ([0, 0], 1));
        assert_eq!(iter.next().unwrap(), ([0, 1], 2));
        assert_eq!(iter.next().unwrap(), ([1, 0], 3));
        assert_eq!(iter.next().unwrap(), ([1, 1], 4));
        assert!(iter.next().is_none());
    }
}
