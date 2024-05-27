use rten::{Dimension, FloatOperators, Model, TensorPool};
use rten_tensor::{AsView, Layout, NdTensor};
use rten::ops::{BoxOrder, non_max_suppression};

pub fn detect_objects(model: &Model, image: &NdTensor<f32, 3>) -> Vec<(u32, u32, u32, u32, i32)> {
    let [_, image_height, image_width] = image.shape();

    let mut image = image.clone().as_dyn().to_tensor();
    image.insert_axis(0);

    let input_shape = model.input_shape(0).expect("Model does not specify expected input shape");
    let (input_h, input_w) = match &input_shape[..] {
        &[_, _, Dimension::Fixed(h), Dimension::Fixed(w)] => (h, w),
        _ => (640, 640),
    };
    let image = image.resize_image([input_h, input_w]).expect("Failed to resize image");
    let input_id = model.node_id("images").expect("Failed to get input node ID");
    let output_id = model.node_id("output0").expect("Failed to get output node ID");

    let [output] = model.run_n(&[(input_id, image.view().into())], [output_id], None)
        .expect("Failed to run model");

    let output: NdTensor<f32, 3> = output.try_into().expect("Failed to convert output tensor");
    let [_batch, box_attrs, _n_boxes] = output.shape();
    assert_eq!(box_attrs, 84);

    let model_in_h = image.size(2);
    let model_in_w = image.size(3);
    let scale_y = image_height as f32 / model_in_h as f32;
    let scale_x = image_width as f32 / model_in_w as f32;

    let boxes = output.slice::<3, _>((.., ..4, ..)).permuted([0, 2, 1]);
    let scores = output.slice::<3, _>((.., 4.., ..));

    let iou_threshold = 10.0;
    let score_threshold = 0.01;

    let nms_boxes = non_max_suppression(
        &TensorPool::new(),
        boxes.view(),
        scores,
        BoxOrder::CenterWidthHeight,
        None,
        iou_threshold,
        score_threshold,
    ).expect("Failed during non-max suppression");

    let [n_selected_boxes, _] = nms_boxes.shape();

    let mut bboxes = Vec::new();

    for b in 0..n_selected_boxes {
        let [batch_idx, cls, box_idx] = nms_boxes.slice(b).to_array();
        let [cx, cy, box_w, box_h] = boxes.slice([batch_idx, box_idx]).to_array();
        let score = scores[[batch_idx as usize, cls as usize, box_idx as usize]];

        let left = (cx - 0.5 * box_w) * scale_x;
        let top = (cy - 0.5 * box_h) * scale_y;
        let right = (cx + 0.5 * box_w) * scale_x;
        let bottom = (cy + 0.5 * box_h) * scale_y;

        let box_width = (right - left) * scale_x;
        let box_height = (bottom - top) * scale_y;

        bboxes.push((left as u32, top as u32, (right - left) as u32 - 1, (bottom - top) as u32 - 1, cls));
    }

    bboxes
}
