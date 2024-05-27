use std::cmp::{max, min};
use std::env;
use std::time::{SystemTime, UNIX_EPOCH};

use ab_glyph::{FontRef, PxScale};
use image::{DynamicImage, GenericImage, GenericImageView, Rgba, RgbaImage};
use rand::prelude::*;
use rten::Model;
use rten_imageio::read_image;
use rten_tensor::{AsView, NdTensor};
use colored::*;
use imageproc::drawing::draw_text_mut;

mod edge_detection;
mod object_detection;
mod glitches;
mod consts;

const MAX_SNIPPET_WIDTH: u32 = 600;
const MAX_SNIPPET_HEIGHT: u32 = 600;

const TEXT_SIZE: f32 = 60.0;

const IMAGE_PATHS: &str = r#"images/_DSC8834.jpg
images/_DSC8913.jpg
images/_DSC8911.jpg
images/_DSC8914.jpg
images/_DSC8915.jpg"#;

const FONT_BYTES: &[u8] = include_bytes!("../RobotoMono-VariableFont_wght.ttf");

fn load_images(image_paths: Vec<&str>) -> (Vec<DynamicImage>, (u32, u32), Vec<NdTensor<f32, 3>>) {
    println!("{}", "Loading images...".green());
    let mut images = Vec::new();
    let mut object_images = Vec::new();
    let mut min_width = u32::MAX;
    let mut min_height = u32::MAX;
    for (i, path) in image_paths.iter().enumerate() {
        println!("{}", format!("Loading image {}: {}", i + 1, path).cyan());
        let img = image::open(path).expect("Failed to open image");
        let object_img = read_image(path).expect("Couldn't open image as tensor!");
        let (width, height) = img.dimensions();
        println!("{}", format!("Image {}: {} ({}x{})", i + 1, path, width, height).yellow());
        min_width = min(min_width, width);
        min_height = min(min_height, height);
        images.push(img);
        object_images.push(object_img);
    }
    println!("{}", "Images loaded successfully.".green());
    (images, (min_width, min_height), object_images)
}

fn setup_rng() -> SmallRng {
    println!("{}", "Setting up random number generator...".green());
    let arg = env::args().collect::<Vec<_>>();
    let seed: [u8; 32] = [arg[1].parse().unwrap(); 32];
    println!("{}", format!("Using seed: {}", arg[1]).cyan());
    SmallRng::from_entropy()
}

fn setup_object_detection() -> Model {
    println!("{}", "Setting up object detection model...".green());
    let model = Model::load_file("yolov8.rten").expect("Couldn't load model!");
    println!("{}", "Object detection model loaded successfully.".green());
    model
}

fn main() {
    println!("{}", "Starting application...".green());
    let mut rng = setup_rng();
    let image_paths: Vec<&str> = IMAGE_PATHS.lines().collect();
    let (images, (min_width, min_height), object_images) = load_images(image_paths);
    let model = setup_object_detection();
    let font = FontRef::try_from_slice(FONT_BYTES).unwrap();

    println!("{}", "Applying edge detection...".blink());
    let mut edge_detected_images = Vec::new();
    for (i, img) in images.iter().enumerate() {
        let edge_img = edge_detection::apply_sobel_edge_detection(img);
        println!("{}", format!("Edge detection applied to image {}", i + 1).yellow());
        edge_detected_images.push(edge_img);
    }
    println!("{}", "Edge detection applied successfully.".green());

    println!("{}", "Applying glitch art transformation...".green());
    let mut glitched_images = Vec::new();
    for (i, (edge_img, tensor_image)) in edge_detected_images.iter().zip(object_images.iter()).enumerate() {
        let glitched_img = glitches::apply_glitch_art_2(&images[i], edge_img, tensor_image, &mut rng, &model, &font);
        println!("{}", format!("Glitch art transformation applied to image {}", i + 1).yellow());
        glitched_images.push(glitched_img);
    }
    println!("{}", "Glitch art transformation applied successfully.".green());

    println!("{}", "Combining glitched images...".green());
    let mut canvas = RgbaImage::new(min_width, min_height);
    for x in 0..min_width {
        for y in 0..min_height {
            let mut r_sum = 0;
            let mut g_sum = 0;
            let mut b_sum = 0;
            let mut a_sum = 255;
            let mut v_p_count = 0;
            for (img, _) in &glitched_images {
                let pixel = img.get_pixel(x % img.width(), y % img.height());
                if pixel != Rgba([0, 0, 0, 255]) {
                    let [r, g, b, a] = pixel.0;
                    r_sum = max(r as u32, r_sum);
                    g_sum = max(g as u32, g_sum);
                    b_sum = max(b as u32, b_sum);
                    a_sum = a as u32;
                    // v_p_count += 1;
                }
            }
            if v_p_count == 0 {
                v_p_count = 1;
            }
            canvas.put_pixel(x, y, Rgba([
                (r_sum / v_p_count) as u8,
                (g_sum / v_p_count) as u8,
                (b_sum/ v_p_count) as u8,
                (a_sum/ v_p_count) as u8,
            ]));
        }
    }
    println!("{}", "Glitched images combined successfully.".green());

    let f_height = TEXT_SIZE;
    let scale = PxScale {
        x: f_height * 1.5,
        y: f_height,
    };

    for (_, labels) in &glitched_images{
        for (label, x, y) in labels{
            draw_text_mut(&mut canvas, Rgba([255, 0, 0, 255]), *x, *y, scale, &font, label);
        }
    }

    println!("{}", "Saving final image...".green());
    let start = SystemTime::now();
    let since_epoch = start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    let filename = format!("output-{}.webp", since_epoch.as_secs());
    canvas.save(&filename).expect("Failed to save image");
    println!("{}", format!("Final image saved as {}", filename).green());
}

