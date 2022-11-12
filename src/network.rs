use itertools::Itertools;
use ndarray::Array1;
use rand::Rng;

use crate::formulas;

#[derive(Debug)]
pub struct Network {
    pub layers: Array1<u32>,
    pub layers_num: usize,
    pub biases: Vec<Array1<f32>>,
    pub weights: Vec<Array1<f32>>,
}

impl Network {
    pub fn new(layers: &[u32]) -> Self {
        Network {
            layers: Array1::from_vec(layers.to_vec()),
            layers_num: layers.len(),
            biases: biases_from_layers(layers),
            weights: weights_from_layers(layers),
        }
    }
}

fn biases_from_layers(layers: &[u32]) -> Vec<Array1<f32>> {
    let mut rng = rand::thread_rng();
    let mut biases: Vec<Array1<f32>> = Vec::new();

    for layer in layers[1..].into_iter() {
        let biases_layer: Vec<f32> = 
        (0..*layer)
        .map(|_| rng.gen_range(0.0..1.0))
        .collect();
        biases.push(Array1::from_vec(biases_layer));
    }

    biases
}

fn weights_from_layers(layers: &[u32]) -> Vec<Array1<f32>> {
    let mut rng = rand::thread_rng();
    let mut weights: Vec<Array1<f32>> = Vec::new();

    for (left, right) in layers.into_iter().tuple_windows() {
        let weights_num: u32 = *left * *right;
        let weights_layer: Vec<f32> = (0..weights_num)
        .map(|_| rng.gen_range(0.0..1.0))
        .collect();
        weights.push(Array1::from_vec(weights_layer));
    }

    weights
}
