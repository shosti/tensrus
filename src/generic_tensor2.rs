use crate::{
    numeric::Numeric,
    shape::Shape,
    storage::{Layout, Storage},
    tensor2::Tensor2,
};

#[derive(Tensor2, Debug, Clone)]
#[tensor_rank = "R"]
#[tensor_shape = "S"]
pub struct GenericTensor2<T: Numeric, const R: usize, const S: Shape> {
    storage: Storage<T>,
    layout: Layout,
}
