mod io;
use self::io::mnist;

// Path to the training data folder
const TRAINING_DATA_FOLDER: &str = "data/train";

fn main() {
    let data: Vec<mnist::Image> = io::get_data(TRAINING_DATA_FOLDER).unwrap();
    for (i, row) in data.into_iter().enumerate() {
        println!("label of image {} is {}", i, row.label);
    }
}
