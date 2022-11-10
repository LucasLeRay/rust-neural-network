use ndarray::Array1;

pub struct Header {
    pub count: usize,
    pub rows_number: Option<i32>,
    pub columns_number: Option<i32>,
}

impl Header {
    pub fn size(&self) -> usize {
        (self.rows_number.unwrap() * self.columns_number.unwrap()) as usize
    }
}

pub struct Data {
    pub header: Header,
    pub content: Vec<u8>
}

#[derive(Debug)]
pub struct Image {
    pub pixels: Array1<f32>,
    pub label: u8,
}
