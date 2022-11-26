use ndarray::{Array, Array1, Array2};
use ndarray_rand::{RandomExt, rand_distr::StandardNormal};
use rand::seq::SliceRandom;

use crate::{formulas, io::mnist::Image, cost::Cost, regularization::Regularization};

#[derive(Debug)]
pub struct Network {
    pub layers: Vec<usize>,
    pub layers_num: usize,
    pub biases: Vec<Array2<f32>>,
    pub weights: Vec<Array2<f32>>,
}

impl Network {
    pub fn new(layers: &[usize]) -> Self {
        let layers_num: usize = layers.len();
        let mut biases: Vec<Array2<f32>> = Vec::new();
        let mut weights: Vec<Array2<f32>> = Vec::new();

        for i in 1..layers_num {
            weights.push(Array::random((layers[i], layers[i - 1]), StandardNormal));
            biases.push(Array::random((layers[i], 1), StandardNormal));
        }

        Network {
            biases,
            weights,
            layers_num,
            layers: layers.to_owned()
        }
    }

    // Perform a Stochastic Gradient Descent to train a neural network.
    pub fn sgd(
        &mut self,
        train_data: &[Image],
        epochs: u32,
        mini_batch_size: usize,
        eta: f32,
        regularization: Option<&dyn Regularization>,
        test_data: &[Image],
        cost: &dyn Cost,
    ) {
        let n_train: usize = train_data.len();
        let n_test: usize = test_data.len();

        for epoch in 0..epochs {
            let mut indices: Vec<usize> = (0..train_data.len()).collect::<Vec<usize>>();
            indices.shuffle(&mut rand::thread_rng());

            for batch_indices in (0..n_train)
                .step_by(mini_batch_size)
                .collect::<Vec<usize>>()
                .windows(2)
            {
                let (start, end) = (batch_indices[0], batch_indices[1]);
                let mini_batch: &[Image] = &train_data[start..end];
                let (gb, gw) = self.gradients_from_mini_batch(mini_batch, cost);

                self.update_biases_from_gradient(gb, eta, mini_batch_size);
                self.update_weights_from_gradient(gw, eta, mini_batch_size);
            }

            println!("Epoch {}: {} / {}", epoch, self.evaluate(test_data), n_test);
        }
    }

    // Compute the gradients of a mini-batch, using the backpropagation.
    fn gradients_from_mini_batch(&self, mini_batch: &[Image], cost: &dyn Cost) -> (Vec<Array2<f32>>, Vec<Array2<f32>>) {
        let mut gradient_b: Vec<Array2<f32>> = zero_vec_like(&self.biases);
        let mut gradient_w: Vec<Array2<f32>> = zero_vec_like(&self.weights);

        for image in mini_batch {
            let (delta_gb, delta_gw) = self.backprop(&image.pixels, image.label, cost);
            for (gb, dgb) in gradient_b.iter_mut().zip(delta_gb.iter()) {
                *gb += dgb
            }
            for (gw, ngw) in gradient_w.iter_mut().zip(delta_gw.iter()) {
                *gw += ngw
            }
        }

        (gradient_b, gradient_w)
    }

    // Perform the backpropagation algorithm for a single training example
    // and return its gradients.
    fn backprop(&self, x: &Array1<f32>, y: u8, cost: &dyn Cost) -> (Vec<Array2<f32>>, Vec<Array2<f32>>) {
        let mut gradient_b: Vec<Array2<f32>> = zero_vec_like(&self.biases);
        let mut gradient_w: Vec<Array2<f32>> = zero_vec_like(&self.weights);

        let (activations, zs): (Vec<Array2<f32>>, Vec<Array2<f32>>) = self.feedforward(&x);

        let mut delta: Array2<f32> = cost.delta(
            &zs[zs.len() - 1],
            &activations[activations.len()-1],
            y as usize
        );
        
        let nbiases: usize = self.biases.len();
        let nweights: usize = self.weights.len();

        gradient_b[nbiases - 1] = formulas::rate_of_change_of_biases(&delta);
        gradient_w[nweights - 1] = formulas::rate_of_change_of_weights(
            &delta, &activations[activations.len()-2]
        );
        
        for l in 2..self.layers_num {
            delta = formulas::layer_error(
                &delta, &self.weights[nweights - l + 1], &zs[zs.len() - l]
            );
            gradient_b[nbiases - l] = formulas::rate_of_change_of_biases(&delta);
            gradient_w[nweights - l] = formulas::rate_of_change_of_weights(
                &delta, &activations[activations.len()-l-1]
            );
        }

        (gradient_b, gradient_w)
    }

    fn update_biases_from_gradient(&mut self, gradients: Vec<Array2<f32>>, eta: f32, batch_size: usize) {
        for (biases_layer, gradient_layer) in self.biases.iter_mut().zip(gradients.iter()) {
            *biases_layer -= &((eta / batch_size as f32) * gradient_layer);
        }
    }

    fn update_weights_from_gradient(&mut self, gradients: Vec<Array2<f32>>, eta: f32, batch_size: usize, n_train: usize, regularization: Option<&dyn Regularization>) {
        for (weights_layer, gradient_layer) in self.weights.iter_mut().zip(gradients.iter()) {
            let scaled_gradient: Array2<f32> = (eta / batch_size as f32) * gradient_layer;
            match regularization {
                None => *weights_layer -= &scaled_gradient,
                Some(reg) => *weights_layer = reg.rescale_weights(&weights_layer, n_train, eta) - scaled_gradient
            }
        }
    }

    // Perform a feedforward on the network from the training example.
    // Return the activations of nodes.
    fn feedforward(&self, x: &Array1<f32>) -> (Vec<Array2<f32>>, Vec<Array2<f32>>) {
        let mut activations: Vec<Array2<f32>> = Vec::new();
        let mut zs: Vec<Array2<f32>> = Vec::new();
        let img_flat_size: usize = x.len();
        activations.push(x.to_shape((img_flat_size, 1)).unwrap().to_owned());

        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            let z: Array2<f32> = w.dot(activations.last().unwrap()) + b;
            zs.push(z);
            let activation: Array2<f32> = formulas::sigmoid(zs.last().unwrap());
            activations.push(activation);
        }

        (activations, zs)
    }

    // Predict label for a single example.
    fn predict(&self, pixels: &Array1<f32>) -> u8 {
        let (activations, _) = self.feedforward(pixels);
        let output: &Array2<f32> = activations.last().unwrap();

        let mut predicted: usize = 0;
        for (i, prediction) in output.iter().enumerate() {
            if *prediction > output[[predicted, 0]] {
                predicted = i;
            }
        }

        predicted as u8
    }

    // Evaluate every example in the testing set and return the number of
    // correct predictions.
    fn evaluate(&self, testing_set: &[Image]) -> u32 {
        let predictions: Vec<u8> = testing_set
            .iter()
            .map(|image| self.predict(&image.pixels))
            .collect();
        testing_set
            .iter()
            .zip(predictions.iter())
            .map(|(image, prediction)| if image.label == *prediction {1} else {0})
            .sum()
    }
}


fn zero_vec_like(vec: &[Array2<f32>]) -> Vec<Array2<f32>>{
    vec.iter().map(|x| {
        let shape = x.shape();
        Array2::zeros((shape[0], shape[1]))
    }).collect()
}
