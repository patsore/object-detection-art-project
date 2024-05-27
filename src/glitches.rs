use std::cmp::min;

use ab_glyph::{FontRef, PxScale};
use image::{DynamicImage, GenericImage, GenericImageView, Rgba};
use image::imageops::overlay;
use imageproc::drawing::{draw_text_mut, text_size};
use rand::prelude::*;
use rten::{Model};
use rten_tensor::NdTensor;

use crate::object_detection::detect_objects;
use crate::consts::LABELS;
use crate::{MAX_SNIPPET_HEIGHT, MAX_SNIPPET_WIDTH, TEXT_SIZE};

pub fn apply_glitch_art_2(img: &DynamicImage, edge_img: &DynamicImage, tensor_image: &NdTensor<f32, 3>, rng: &mut SmallRng, model: &Model, font: &FontRef) -> (DynamicImage, Vec<(String, i32, i32)>) {
    let object_boxes = detect_objects(model, tensor_image);

    let (a_width, a_height) = img.dimensions();
    let edge_data: Vec<u8> = edge_img.to_luma8().into_raw();
    let mut result = DynamicImage::ImageRgba8(edge_img.clone().to_rgba8());

    let mut labels = Vec::new();

    for (x, y, mut width, mut height, cls) in object_boxes {
        if x >= a_width || y >= a_height {
            continue;
        }
        width = min(MAX_SNIPPET_WIDTH, min(width, a_width - x));
        height = min(MAX_SNIPPET_HEIGHT, min(height, a_height - y));
        let snippet_image = img.view(x, y, width, height);
        overlay(&mut result, &snippet_image.to_image(), x as i64, y as i64);

        let label = LABELS[cls as usize];
        let f_height = TEXT_SIZE;
        let scale = PxScale {
            x: f_height * 1.5,
            y: f_height,
        };
        let offset = text_size(scale, font, label);
        labels.push((label.to_string(), (x + width / 2) as i32 - offset.0 as i32 / 2, (y + height / 2) as i32 - offset.1 as i32 / 2))
    }

    (result, labels)
}

fn get_luminance(pixel: Rgba<u8>) -> u8 {
    let [r, g, b, _a] = pixel.0;
    (0.2126 * r as f64 + 0.7152 * g as f64 + 0.0722 * b as f64) as u8
}
