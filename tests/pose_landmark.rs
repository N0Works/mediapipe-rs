use mediapipe_rs::tasks::vision::PoseLandmarkerBuilder;

const MODEL_PATH: &'static str = "assets/models/pose_landmark/pose_landmarker_heavy.task";
const BODY_1: &'static str = "assets/testdata/img/full_body.jpg";

#[test]
fn test_pose_landmark() {
    let img = image::open(BODY_1).unwrap();
    let pose_landmark_results = PoseLandmarkerBuilder::new()
        .cpu()
        .num_poses(1)
        .output_segmentation_masks(true)
        .build_from_file(MODEL_PATH)
        .unwrap()
        .detect(&img)
        .unwrap();

    assert!(!pose_landmark_results.is_empty());
    let first = &pose_landmark_results[0];
    assert_eq!(first.pose_landmarks.len(), 33);
    assert_eq!(first.pose_world_landmarks.len(), 33);
    assert!(first.segmentation_mask.is_some());
    eprintln!("{}", pose_landmark_results);
}
