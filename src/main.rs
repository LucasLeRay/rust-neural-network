mod io;
mod network;

use crate::network::Network;
use crate::io::mnist::Image;
mod formulas;

const TRAINING_DATA_FOLDER: &str = "data/train";

fn main() {
    let data: Vec<Image> = io::get_data(TRAINING_DATA_FOLDER).unwrap();
    for (i, row) in data.into_iter().enumerate() {
        println!("label of image {} is {}", i, row.label);
    }

    let network = Network::new(&[2, 3, 1]);
    println!("{:?}", network);
}
