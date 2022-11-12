use rand::Rng;

use ndarray::Array1;

#[derive(Debug)]
pub struct Network {
    pub size: Array1<u32>,
    pub layers_num: usize,
    pub biases: Vec<Array1<f32>>,
    pub weights: Vec<Array1<f32>>,
}

impl Network {
    pub fn new(sizes: &[u32]) -> Self {
        let mut rng = rand::thread_rng();

        let mut biases: Vec<Array1<f32>> = Vec::new();
        let mut weights: Vec<Array1<f32>> = Vec::new();

        for layer in sizes[1..].into_iter() {
            let biases_layer: Vec<f32> = 
                (0..*layer)
                .map(|_| rng.gen_range(0.0..1.0))
                .collect();
            biases.push(Array1::from_vec(biases_layer));
        }

        for (i, layer) in sizes[0..(sizes.len() - 1)].into_iter().enumerate() {
            let weights_num: u32 = *layer * sizes[i + 1];
            let weights_layer: Vec<f32> = (0..weights_num)
            .map(|_| rng.gen_range(0.0..1.0))
            .collect();
            weights.push(Array1::from_vec(weights_layer));
        }

        Network {
            size: Array1::from_vec(sizes.to_vec()),
            layers_num: sizes.len(),
            biases,
            weights,
        }
    }
}
