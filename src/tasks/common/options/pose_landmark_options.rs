#[derive(Clone)]
pub(crate) struct PoseLandmarkOptions {
    /// The maximum number of poses can be detected by the PoseLandmarker.
    pub num_poses: i32,

    /// The minimum confidence score for the pose detection to be considered successful.
    pub min_pose_detection_confidence: f32,

    /// The minimum confidence score of pose presence score in the pose landmark detection.
    pub min_pose_presence_confidence: f32,

    /// The minimum confidence score for the pose tracking to be considered successful.
    pub min_tracking_confidence: f32,

    /// Whether PoseLandmarker outputs segmentation masks.
    pub output_segmentation_masks: bool,
}

impl Default for PoseLandmarkOptions {
    #[inline(always)]
    fn default() -> Self {
        Self {
            num_poses: 1,
            min_pose_detection_confidence: 0.5,
            min_pose_presence_confidence: 0.5,
            min_tracking_confidence: 0.5,
            output_segmentation_masks: false,
        }
    }
}

macro_rules! pose_landmark_options_impl {
    () => {
        /// Set the maximum number of poses can be detected by the PoseLandmarker.
        #[inline(always)]
        pub fn num_poses(mut self, num_poses: i32) -> Self {
            self.pose_landmark_options.num_poses = num_poses;
            self
        }

        /// Set the minimum confidence score for the pose detection to be considered successful.
        #[inline(always)]
        pub fn min_pose_detection_confidence(mut self, min_pose_detection_confidence: f32) -> Self {
            self.pose_landmark_options.min_pose_detection_confidence =
                min_pose_detection_confidence;
            self
        }

        /// Set the minimum confidence score of pose presence score in the pose landmark detection.
        #[inline(always)]
        pub fn min_pose_presence_confidence(mut self, min_pose_presence_confidence: f32) -> Self {
            self.pose_landmark_options.min_pose_presence_confidence = min_pose_presence_confidence;
            self
        }

        /// Set the minimum confidence score for the pose tracking to be considered successful.
        #[inline(always)]
        pub fn min_tracking_confidence(mut self, min_tracking_confidence: f32) -> Self {
            self.pose_landmark_options.min_tracking_confidence = min_tracking_confidence;
            self
        }

        /// Set whether PoseLandmarker outputs segmentation masks.
        #[inline(always)]
        pub fn output_segmentation_masks(mut self, output_segmentation_masks: bool) -> Self {
            self.pose_landmark_options.output_segmentation_masks = output_segmentation_masks;
            self
        }
    };
}

macro_rules! pose_landmark_options_check {
    ( $self:ident ) => {{
        if $self.pose_landmark_options.num_poses == 0 {
            return Err(crate::Error::ArgumentError(
                "The number of max poses cannot be zero".into(),
            ));
        }
        if $self.pose_landmark_options.min_pose_presence_confidence < 0.
            || $self.pose_landmark_options.min_pose_presence_confidence > 1.
        {
            return Err(crate::Error::ArgumentError(format!(
                "The min_pose_presence_confidence must in range [0.0, 1.0], but got `{}`",
                $self.pose_landmark_options.min_pose_presence_confidence
            )));
        }
        if $self.pose_landmark_options.min_pose_detection_confidence < 0.
            || $self.pose_landmark_options.min_pose_detection_confidence > 1.
        {
            return Err(crate::Error::ArgumentError(format!(
                "The min_pose_detection_confidence must in range [0.0, 1.0], but got `{}`",
                $self.pose_landmark_options.min_pose_detection_confidence
            )));
        }
        if $self.pose_landmark_options.min_tracking_confidence < 0.
            || $self.pose_landmark_options.min_tracking_confidence > 1.
        {
            return Err(crate::Error::ArgumentError(format!(
                "The min_tracking_confidence must in range [0.0, 1.0], but got `{}`",
                $self.pose_landmark_options.min_tracking_confidence
            )));
        }
    }};
}

macro_rules! pose_landmark_options_get_impl {
    () => {
        /// Get the maximum number of poses can be detected by the PoseLandmarker.
        #[inline(always)]
        pub fn num_poses(&self) -> i32 {
            self.build_options.pose_landmark_options.num_poses
        }

        /// Get the minimum confidence score for the pose detection to be considered successful.
        #[inline(always)]
        pub fn min_pose_detection_confidence(&self) -> f32 {
            self.build_options
                .pose_landmark_options
                .min_pose_detection_confidence
        }

        /// Get the minimum confidence score of pose presence score in the pose landmark detection.
        #[inline(always)]
        pub fn min_pose_presence_confidence(&self) -> f32 {
            self.build_options
                .pose_landmark_options
                .min_pose_presence_confidence
        }

        /// Get the minimum confidence score for the pose tracking to be considered successful.
        #[inline(always)]
        pub fn min_tracking_confidence(&self) -> f32 {
            self.build_options
                .pose_landmark_options
                .min_tracking_confidence
        }

        /// Get whether PoseLandmarker outputs segmentation masks.
        #[inline(always)]
        pub fn output_segmentation_masks(&self) -> bool {
            self.build_options
                .pose_landmark_options
                .output_segmentation_masks
        }
    };
}
