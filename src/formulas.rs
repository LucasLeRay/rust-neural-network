// This module contains scientific formulas. It could be replaced
// by the usage of a scientific library.

use libm::expf;
use ndarray::Array1;

pub fn sigmoid(z: &Array1<f32>) -> Array1<f32> {
    z.map(|x| 1. / (1. + expf(-x)))
}

pub fn sigmoid_prime(z: &Array1<f32>) -> Array1<f32> {
    sigmoid(z) * (1.0 - sigmoid(z))
}
