use std::{fs::File, io::{Write, BufReader}};

use crate::network::Network;

pub fn save(network: &Network, filepath: &str) {
    let content: String = serde_json::to_string(network).expect("Unable to serialize model.");
    let mut file: File = File::create(filepath).expect("Unable to create model file.");
    file.write_all(content.to_string().as_bytes()).expect("Unable to write model into file.");
}

pub fn load(filepath: &str) -> Network {
    let file: File = File::open(filepath).expect("Unable to open model file.");
    let reader: BufReader<File> = BufReader::new(file);
    serde_json::from_reader(reader).expect("Unable to parse model file.")
}
