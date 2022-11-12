// This module contains scientific formulas. It could be replaced
// by the usage of a scientific library.

use libm::expf;
use ndarray::Array1;

pub fn sigmoid(z: &Array1<f32>) -> Array1<f32> {
    z.clone().map(|x| 1. / (1. + expf(-x)))
}
