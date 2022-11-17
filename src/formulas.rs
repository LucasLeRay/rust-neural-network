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

// compute the error vector of the output layer from the cost and z.
pub fn output_error(cost: &Array2<f32>, z: &Array2<f32>) -> Array2<f32> {
    cost * sigmoid_prime(z)
}

// compute the error vector of a layer, from the error, weight matrix and z
// of the next layer.
pub fn layer_error(delta: &Array2<f32>, weights: &Array2<f32>, z: &Array2<f32>) -> Array2<f32> {
    weights.t().dot(delta) * sigmoid_prime(z)
}

// compute the rate of change of the cost with respect to the biases
// of a layer, from the delta of this layer.
// since the error of a layer equals exactly the rate of change, delta is
// simply copied.
pub fn rate_of_change_of_biases(delta: &Array2<f32>) -> Array2<f32> {
    delta.to_owned()
}

// compute the rate of change of the cost with respect to the weights
// of its layer, from the delta of the next layer and the activations of
// the previous one.
pub fn rate_of_change_of_weights(delta: &Array2<f32>, activations: &Array2<f32>) -> Array2<f32> {
    delta.dot(&activations.t())
}
