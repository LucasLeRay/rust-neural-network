// This module contains scientific formulas. It could be replaced
// by the usage of a scientific library.

use libm::expf;
use ndarray::Array2;

pub fn sigmoid(z: &Array2<f32>) -> Array2<f32> {
    z.map(|x| 1. / (1. + expf(-x)))
}

pub fn sigmoid_prime(z: &Array2<f32>) -> Array2<f32> {
    sigmoid(z) * (1.0 - sigmoid(z))
}
