use ndarray::Array2;

pub trait Regularization {
    fn rescale_weights(&self, w: &Array2<f32>, n: usize, eta: f32) -> Array2<f32>;
}

pub struct L2 {
    pub lambda: f32
}

impl Regularization for L2 {
    fn rescale_weights(&self, w: &Array2<f32>, n: usize, eta: f32) -> Array2<f32>{
        w * (1.0 - (eta * self.lambda) / n as f32)
    }
}

pub struct L1 {
    pub lambda: f32
}

impl Regularization for L1 {
    fn rescale_weights(&self, w: &Array2<f32>, n: usize, eta: f32) -> Array2<f32> {
        let w_sign: Array2<f32> = w.map(|x| {
            if *x < 0. {-1.} else if *x > 0. {1.} else {0.}
        });
        w - ((eta * self.lambda) / n as f32) * w_sign
    }
}
