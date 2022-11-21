use lazy_static::lazy_static;
use ndarray::{Array2, array};

pub trait Cost {
    fn derivative(&self, a: &Array2<f32>, y: usize) -> Array2<f32>;
}

pub struct QuadraticCost;

impl Cost for QuadraticCost {
    fn derivative(&self, a: &Array2<f32>, y: usize) -> Array2<f32> {
        a - ACTIVATIONS[y].to_owned()
    }
}

lazy_static! {
    static ref ACTIVATIONS: [Array2<f32>; 10] = [
        array![[1.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]],
        array![[0.], [1.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]],
        array![[0.], [0.], [1.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]],
        array![[0.], [0.], [0.], [1.], [0.], [0.], [0.], [0.], [0.], [0.]],
        array![[0.], [0.], [0.], [0.], [1.], [0.], [0.], [0.], [0.], [0.]],
        array![[0.], [0.], [0.], [0.], [0.], [1.], [0.], [0.], [0.], [0.]],
        array![[0.], [0.], [0.], [0.], [0.], [0.], [1.], [0.], [0.], [0.]],
        array![[0.], [0.], [0.], [0.], [0.], [0.], [0.], [1.], [0.], [0.]],
        array![[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [1.], [0.]],
        array![[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [1.]]
    ];
}


