use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum IndexError {
    OutOfBounds,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GraphError {
    GraphDropped,
}

#[derive(Debug)]
pub enum UpdateFromGradError {
    GradNotCalculated,
}

impl Display for IndexError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            Self::OutOfBounds => {
                write!(f, "out of bounds")
            }
        }
    }
}
impl std::error::Error for IndexError {}

impl Display for GraphError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            Self::GraphDropped => {
                write!(f, "attempted to perform graph operations on a non-root node after the root node was dropped")
            }
        }
    }
}

impl std::error::Error for GraphError {}

impl Display for UpdateFromGradError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            Self::GradNotCalculated => {
                write!(
                    f,
                    "attempted to call update_from_grad() before grad calculated with backward()"
                )
            }
        }
    }
}
impl std::error::Error for UpdateFromGradError {}
