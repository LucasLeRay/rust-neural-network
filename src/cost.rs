use lazy_static::lazy_static;
use ndarray::{Array2, array};
use ndarray_linalg::Norm;

use crate::formulas;

pub trait Cost {
    fn delta(&self, z: &Array2<f32>, a: &Array2<f32>, y: usize) -> Array2<f32>;

    fn sum(&self, a: &Array2<f32>, y: usize) -> f32;
}

pub struct QuadraticCost;

impl Cost for QuadraticCost {
    fn sum(&self, a: &Array2<f32>, y: usize) -> f32 {
        (a - ACTIVATIONS[y].to_owned()).norm()
    }

    fn delta(&self, z: &Array2<f32>, a: &Array2<f32>, y: usize) -> Array2<f32> {
        (a - ACTIVATIONS[y].to_owned()) * formulas::sigmoid_prime(z)
    }
}

pub struct CrossEntropyCost;

impl Cost for CrossEntropyCost {
    fn sum(&self, a: &Array2<f32>, y: usize) -> f32 {
        let y = ACTIVATIONS[y].to_owned();

        let a_log: Array2<f32> = a.mapv(|x| x.ln());
        let one_minus_a_log: Array2<f32> = a.mapv(|x| 1. - x.ln());
        let costs = -&y * a_log - (1. - &y) * one_minus_a_log;

        costs.map(|x| if x.is_nan() {0.} else {*x}).sum()
    }

    fn delta(&self, _z: &Array2<f32>, a: &Array2<f32>, y: usize) -> Array2<f32> {
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


