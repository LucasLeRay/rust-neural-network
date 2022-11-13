use itertools::Itertools;
use ndarray::{Array1, Array2};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use rand::{seq::SliceRandom};

use crate::{formulas, io::mnist::Image};

#[derive(Debug)]
pub struct Network {
    pub layers: Vec<u32>,
    pub layers_num: usize,
    pub biases: Vec<Array1<f32>>,
    pub weights: Vec<Array2<f32>>,
}

impl Network {
    pub fn new(layers: &[u32]) -> Self {
        let layers: Vec<u32> = layers.to_vec();

        Network {
            biases: biases_from_layers(&layers, true),
            weights: weights_from_layers(&layers, true),
            layers_num: layers.len(),
            layers,
        }
    }

    pub fn sgd(
        &mut self,
        train_data: &mut Vec<Image>,
        epochs: u32,
        mini_batch_size: usize,
        eta: f32,
    ) {
        let n = train_data.len();

        for _epoch in 0..epochs {
            train_data.shuffle(&mut rand::thread_rng());

            for i in (0..n).step_by(mini_batch_size) {
                let mini_batch: &[Image] = &train_data[i..(i+mini_batch_size)];
                let (gb, gw) = self.gradients_from_mini_batch(mini_batch);

                self.update_biases_from_gradient(gb, eta, mini_batch_size);
                self.update_weights_from_gradient(gw, eta, mini_batch_size);
            }
        }
    }

    fn gradients_from_mini_batch(&self, mini_batch: &[Image]) -> (Vec<Array1<f32>>, Vec<Array2<f32>>) {
        let mut gradients_b: Vec<Array1<f32>> = biases_from_layers(&self.layers, false);
        let mut gradients_w: Vec<Array2<f32>> = weights_from_layers(&self.layers, false);

        for image in mini_batch {
            let (delta_gb, delta_gw) = self.backprop(&image.pixels, image.label);
            for (layer_gw, dgw) in gradients_w.iter_mut().zip(delta_gw.iter()) {
                *layer_gw += dgw
            }
            for (layer_gb, dgb) in gradients_b.iter_mut().zip(delta_gb.iter()) {
                *layer_gb += dgb
            }
        }

        (gradients_b, gradients_w)
    }

    fn backprop(&self, x: &Array1<f32>, y: u8) -> (Vec<Array1<f32>>, Vec<Array2<f32>>) {
        let mut delta_gb: Vec<Array1<f32>> = biases_from_layers(&self.layers, false);
        let mut delta_gw: Vec<Array2<f32>> = weights_from_layers(&self.layers, false);

        let activations: Vec<Array1<f32>> = self.feedforward(x.clone());

        // TODO: backward propagation
                
        (delta_gb, delta_gw)
    }
            
    fn update_biases_from_gradient(&mut self, gradients: Vec<Array1<f32>>, eta: f32, batch_size: usize) {
        for (biases_layer, gradient_layer) in self.biases.iter_mut().zip(gradients.iter()) {
            *biases_layer -= &((eta / batch_size as f32) * gradient_layer);
        }
    }
    
    fn update_weights_from_gradient(&mut self, gradients: Vec<Array2<f32>>, eta: f32, batch_size: usize) {
        for (weights_layer, gradient_layer) in self.weights.iter_mut().zip(gradients.iter()) {
            *weights_layer -= &((eta / batch_size as f32) * gradient_layer);
        }
    }

    // Compute the cost of the output layer from the expected number.
    fn cost(&self, output: &Array1<f32>, y: u8) -> Array1<f32> {
        let mut y_arr: Array1<f32> = Array1::zeros(10);
        y_arr[y as usize] = 1.0;
        
        output - y_arr
    }

    // Perform a feedforward on the network from the training example.
    // Return the activations of nodes.
    fn feedforward(&self, x: Array1<f32>) -> Vec<Array1<f32>> {
        let mut activations: Vec<Array1<f32>> = Vec::new();
        activations.push(x.clone());

        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            let z = activations.last().unwrap().dot(w) + b;
            activations.push(formulas::sigmoid(&z));
        }

        activations
    }
}

// Create a vector of arrays containing values related to weights.
fn biases_from_layers(layers: &Vec<u32>, random: bool) -> Vec<Array1<f32>> {
    let mut rng = rand::thread_rng();
    let mut biases: Vec<Array1<f32>> = Vec::new();

    for layer in layers[1..].into_iter() {
        let shape: usize = *layer as usize;
        let layer: Array1<f32> = if random {
            Array1::random_using(shape, Uniform::new(0., 1.), &mut rng)
        } else {
            Array1::zeros(shape as usize)
        };
        biases.push(layer);
    }

    biases
}

// Create a vector of arrays containing values related to biases.
fn weights_from_layers(layers: &Vec<u32>, random: bool) -> Vec<Array2<f32>> {
    let mut rng = rand::thread_rng();
    let mut weights: Vec<Array2<f32>> = Vec::new();
    
    for (left, right) in layers.into_iter().tuple_windows() {
        let shape: (usize, usize) = (*left as usize, *right as usize);
        let layer: Array2<f32> = if random {
            Array2::random_using(shape, Uniform::new(0., 1.), &mut rng)
        } else {
            Array2::zeros(shape)
        };
        weights.push(layer);
    }

    weights
}
