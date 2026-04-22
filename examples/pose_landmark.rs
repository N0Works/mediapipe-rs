fn parse_args() -> Result<(String, String, Option<String>), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 && args.len() != 4 {
        return Err(format!(
            "Usage {} model_path image_path [output image path]",
            args[0]
        )
        .into());
    }
    Ok((args[1].clone(), args[2].clone(), args.get(3).cloned()))
}

use mediapipe_rs::postprocess::utils::DrawLandmarksOptions;
use mediapipe_rs::tasks::vision::{PoseLandmark, PoseLandmarkerBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (model_path, img_path, output_path) = parse_args()?;

    let mut input_img = image::open(img_path)?;
    let pose_landmark_results = PoseLandmarkerBuilder::new()
        .num_poses(1)
        .min_pose_detection_confidence(0.5)
        .min_pose_presence_confidence(0.5)
        .min_tracking_confidence(0.5)
        .build_from_file(model_path)?
        .detect(&input_img)?;

    println!("{}", pose_landmark_results);

    if let Some(output_path) = output_path {
        let options = DrawLandmarksOptions::default()
            .connections(PoseLandmark::CONNECTIONS)
            .landmark_radius_percent(0.006);
        for result in pose_landmark_results.iter() {
            result.draw_with_options(&mut input_img, &options);
        }
        input_img.save(output_path)?;
    }

    Ok(())
}
