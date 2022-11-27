mod cost;
mod io;
mod formulas;
mod network;
mod regularization;

use chrono::Utc;
use regularization::L2;
use cost::CrossEntropyCost;

use crate::network::Network;
use crate::io::mnist::Image;

const TRAINING_DATA_FOLDER: &str = "data/train";
const TESTING_DATA_FOLDER: &str = "data/test";
const MODEL_FOLDER: &str = "data/artefacts";

fn main() {
    let mut train_data: Vec<Image> = io::data::get_data(TRAINING_DATA_FOLDER).unwrap();
    let test_data: Vec<Image> = io::data::get_data(TESTING_DATA_FOLDER).unwrap();
    let mut network: Network = Network::new(&[784, 100, 10]);

    let cost: CrossEntropyCost = cost::CrossEntropyCost;
    let regularization: L2 = L2 {lambda: 5.0};

    network.sgd(
        &mut train_data,
        100,
        10,
        0.5,
        10,
        Some(&regularization),
        &test_data,
        &cost,
    );

    let model_filepath: String = format!("{}/{}.json", MODEL_FOLDER, Utc::now());
    network.save(&model_filepath);
}
