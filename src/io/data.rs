use std::{fs::File, io::{Read, Cursor}};

use byteorder::{ReadBytesExt, BigEndian};
use flate2::read::GzDecoder;
use ndarray::Array1;

use crate::io::mnist;

const MAGIC_NUMBER_IMAGES_FILE: i32 = 2051;
const MAGIC_NUMBER_LABELS_FILE: i32 = 2049;

fn read_content(f: File) -> Result<Vec<u8>, std::io::Error> {
    let mut gz: GzDecoder<File> = GzDecoder::new(f);

    let mut buf: Vec<u8> = Vec::new();
    gz.read_to_end(&mut buf)?;

    Ok(buf)
}

fn parse_images_header(r: &mut Cursor<&Vec<u8>>) -> Result<mnist::Header, std::io::Error> {
    let count: usize = r.read_i32::<BigEndian>()? as usize;

    let rows_number: i32 = r.read_i32::<BigEndian>()?;
    let columns_number: i32 = r.read_i32::<BigEndian>()?;

    Ok(mnist::Header{
        count,
        rows_number: Some(rows_number),
        columns_number: Some(columns_number)
    })
}

fn parse_labels_header(r: &mut Cursor<&Vec<u8>>) -> Result<mnist::Header, std::io::Error> {
    let count: usize = r.read_i32::<BigEndian>()? as usize;

    Ok(mnist::Header{
        count,
        rows_number: None,
        columns_number: None,
    })
}

fn parse_data(buf: Vec<u8>) -> Result<mnist::Data, std::io::Error> {
    let mut r: Cursor<&Vec<u8>> = Cursor::new(&buf);

    let magic_number: i32 = r.read_i32::<BigEndian>()?;
    let header: mnist::Header = match magic_number {
        MAGIC_NUMBER_IMAGES_FILE => parse_images_header(&mut r)?,
        MAGIC_NUMBER_LABELS_FILE => parse_labels_header(&mut r)?,
        _ => panic!("Invalid magic number, got: {}", magic_number)
    };

    let mut content: Vec<u8> = Vec::new();
    r.read_to_end(&mut content)?;

    Ok(mnist::Data{header, content})
}

fn build_dataset(images: mnist::Data, labels: mnist::Data) -> Vec<mnist::Image> {
    let mut images_pixels: Vec<Array1<f32>> = Vec::new();
    for i in 0..images.header.count {
        let size: usize = images.header.size();
        let start: usize = i * size;
        let end: usize = start + size;
        let image_pixels: Vec<u8> = images.content[start..end].to_vec();
        let rescaled_pixels: Vec<f32> = image_pixels.into_iter().map(|x| x as f32 / 255.).collect();
        images_pixels.push(Array1::from_shape_vec(size, rescaled_pixels).unwrap());
    }

    let mut dataset: Vec<mnist::Image> = Vec::new();
    for (pixels, label) in images_pixels.into_iter().zip(labels.content.into_iter()) {
        dataset.push(mnist::Image { pixels, label })
    }

    dataset
}

pub fn get_data(data_folder_path: &str) -> Result<Vec<mnist::Image>, std::io::Error> {
    let images_filepath: String = format!("{}/images.gz", data_folder_path);
    let images_buffer: Vec<u8> = read_content(File::open(images_filepath)?)?;
    let images_data: mnist::Data = parse_data(images_buffer)?;

    let labels_filepath: String = format!("{}/labels.gz", data_folder_path);
    let labels_buffer: Vec<u8> = read_content(File::open(labels_filepath)?)?;
    let labels_data: mnist::Data = parse_data(labels_buffer)?;

    let dataset: Vec<mnist::Image> = build_dataset(images_data, labels_data);

    Ok(dataset)
}
