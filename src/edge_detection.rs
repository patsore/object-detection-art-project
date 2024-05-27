use image::{DynamicImage, Luma};
use imageproc::gradients::sobel_gradients;

pub fn apply_sobel_edge_detection(img: &DynamicImage) -> DynamicImage {
    let gray_image = img.to_luma8();
    let grad_x = sobel_gradients(&gray_image);
    let grad_y = sobel_gradients(&gray_image);

    let mut sobel_image = gray_image.clone();
    for (x, y, pixel) in sobel_image.enumerate_pixels_mut() {
        let gx = grad_x.get_pixel(x, y)[0] as i32;
        let gy = grad_y.get_pixel(x, y)[0] as i32;
        let magnitude = ((gx * gx + gy * gy) as f64).sqrt() as u8;
        if magnitude < 40 {
            *pixel = Luma([0]);
            continue
        }
        *pixel = Luma([magnitude]);
    }

    DynamicImage::ImageLuma8(sobel_image)
}
