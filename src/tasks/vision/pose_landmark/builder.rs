use super::{PoseDetector, PoseLandmarker, TensorType};

use crate::model::ZipFiles;
use crate::tasks::common::{BaseTaskOptions, PoseLandmarkOptions};

/// Configure the build options of a new **Pose Landmark** task instance.
///
/// Methods can be chained on it in order to configure it.
pub struct PoseLandmarkerBuilder {
    pub(in super::super) base_task_options: BaseTaskOptions,
    pub(in super::super) pose_landmark_options: PoseLandmarkOptions,
}

impl Default for PoseLandmarkerBuilder {
    #[inline(always)]
    fn default() -> Self {
        Self {
            base_task_options: Default::default(),
            pose_landmark_options: Default::default(),
        }
    }
}

impl PoseLandmarkerBuilder {
    /// Create a new builder with default options.
    #[inline(always)]
    pub fn new() -> Self {
        Default::default()
    }

    base_task_options_impl!(PoseLandmarker);

    pose_landmark_options_impl!();

    pub const POSE_DETECTOR_CANDIDATE_NAMES: &'static [&'static str] = &["pose_detector.tflite"];
    pub const POSE_LANDMARKS_CANDIDATE_NAMES: &'static [&'static str] =
        &["pose_landmarks_detector.tflite"];

    /// Use the current build options and use the buffer as model data to create a new task instance.
    #[inline]
    pub fn build_from_buffer(
        self,
        buffer: impl AsRef<[u8]>,
    ) -> Result<PoseLandmarker, crate::Error> {
        pose_landmark_options_check!(self);
        let buf = buffer.as_ref();

        let zip_file = ZipFiles::new(buf)?;
        let pose_detector_file = search_file_in_zip!(
            zip_file,
            buf,
            Self::POSE_DETECTOR_CANDIDATE_NAMES,
            "PoseDetection"
        );
        let pose_landmark_file = search_file_in_zip!(
            zip_file,
            buf,
            Self::POSE_LANDMARKS_CANDIDATE_NAMES,
            "PoseLandmark"
        );

        let pose_detector = PoseDetector::new(
            self.base_task_options.device,
            self.pose_landmark_options.num_poses,
            self.pose_landmark_options.min_pose_detection_confidence,
            pose_detector_file,
        )?;

        let model_resource = crate::model::parse_model(pose_landmark_file.as_ref())?;
        model_base_check_impl!(model_resource, 1, 5);
        model_resource_check_and_get_impl!(model_resource, to_tensor_info, 0).try_to_image()?;
        let input_tensor_type =
            model_resource_check_and_get_impl!(model_resource, input_tensor_type, 0);

        let score_buf_index = 1;
        let landmarks_buf_index = 0;
        let segmentation_buf_index = 2;
        let world_landmarks_buf_index = 4;

        check_tensor_type!(
            model_resource,
            score_buf_index,
            output_tensor_type,
            TensorType::F32
        );

        let graph = crate::GraphBuilder::new(
            model_resource.model_backend(),
            self.base_task_options.device,
        )
        .build_from_bytes([pose_landmark_file])?;

        Ok(PoseLandmarker {
            build_options: self,
            model_resource,
            graph,
            pose_detector,
            score_buf_index,
            landmarks_buf_index,
            segmentation_buf_index,
            world_landmarks_buf_index,
            input_tensor_type,
        })
    }
}
