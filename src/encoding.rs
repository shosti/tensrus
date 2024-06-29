#[derive(Debug, PartialEq)]
pub enum EncodingError {
    InvalidOneHotInput,
}

impl std::fmt::Display for EncodingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            Self::InvalidOneHotInput => {
                write!(f, "invalid one-hot encoding input")
            }
        }
    }
}
impl std::error::Error for EncodingError {}
