mod io;
mod network;

use crate::network::Network;
use crate::io::mnist::Image;
mod formulas;

const TRAINING_DATA_FOLDER: &str = "data/train";

fn main() {
    let mut data: Vec<Image> = io::get_data(TRAINING_DATA_FOLDER).unwrap();
    let mut network = Network::new(&[784, 30, 10]);

    network.sgd(&mut data, 1, 10, 0.00001);
}
