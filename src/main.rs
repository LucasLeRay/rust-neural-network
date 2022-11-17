mod io;
mod network;

use crate::network::Network;
use crate::io::mnist::Image;
mod formulas;

const TRAINING_DATA_FOLDER: &str = "data/train";
const TESTING_DATA_FOLDER: &str = "data/test";

fn main() {
    let mut train_data: Vec<Image> = io::get_data(TRAINING_DATA_FOLDER).unwrap();
    let test_data: Vec<Image> = io::get_data(TESTING_DATA_FOLDER).unwrap();
    let mut network = Network::new(&[784, 30, 10]);

    network.sgd(&mut train_data, 30, 10, 0.00001, &test_data);
}
