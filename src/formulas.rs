// This module contains scientific formulas. It could be replaced
// by the usage of a scientific library.

use libm::expf;

pub fn sigmoid(z: f32) -> f32 {
    1. / (1. + expf(-z))
}
