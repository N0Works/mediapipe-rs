mod builder;
mod pose_landmark;
mod result;

pub use builder::PoseLandmarkerBuilder;
pub use pose_landmark::PoseLandmark;
pub use result::{PoseLandmarkResult, PoseLandmarkResults};

use crate::model::ModelResourceTrait;
use crate::postprocess::{
    projection_normalized_landmarks, projection_world_landmark, Anchor, CategoriesFilter,
    DetectionBoxFormat, DetectionResult, ImageConfidenceMask, Landmarks,
    NonMaxSuppressionAlgorithm, NonMaxSuppressionOverlapType, NormalizedRect, TensorsToDetection,
    TensorsToLandmarks, TensorsToSegmentation, VideoResultsIter,
};
use crate::preprocess::vision::{ImageToTensor, ImageToTensorInfo, VideoData};
use crate::{Device, Error, Graph, GraphExecutionContext, TensorType};

/// Performs pose landmark on images and video frames.
pub struct PoseLandmarker {
    build_options: PoseLandmarkerBuilder,
    model_resource: Box<dyn ModelResourceTrait>,
    graph: Graph,

    pose_detector: PoseDetector,

    score_buf_index: usize,
    landmarks_buf_index: usize,
    segmentation_buf_index: usize,
    world_landmarks_buf_index: usize,

    input_tensor_type: TensorType,
}

impl PoseLandmarker {
    const RAW_LANDMARKS_NUM: usize = 39;
    const LANDMARKS_NORMALIZE_Z: f32 = 0.4;

    detector_impl!(PoseLandmarkerSession, PoseLandmarkResults);

    pose_landmark_options_get_impl!();

    /// Create a new task session that contains processing buffers and can do inference.
    #[inline(always)]
    pub fn new_session(&self) -> Result<PoseLandmarkerSession, Error> {
        let image_to_tensor_info =
            model_resource_check_and_get_impl!(self.model_resource, to_tensor_info, 0)
                .try_to_image()?;
        let input_tensor_shape =
            model_resource_check_and_get_impl!(self.model_resource, input_tensor_shape, 0);

        let landmarks_out =
            get_type_and_quantization!(self.model_resource, self.landmarks_buf_index);
        let landmarks_shape = model_resource_check_and_get_impl!(
            self.model_resource,
            output_tensor_shape,
            self.landmarks_buf_index
        );
        let mut tensors_to_landmarks =
            TensorsToLandmarks::new(Self::RAW_LANDMARKS_NUM, landmarks_out, landmarks_shape)?;
        tensors_to_landmarks
            .set_image_size(image_to_tensor_info.width(), image_to_tensor_info.height());
        tensors_to_landmarks.set_normalize_z(Self::LANDMARKS_NORMALIZE_Z);
        tensors_to_landmarks.set_visibility_score_sigmoid(true);
        tensors_to_landmarks.set_presence_score_sigmoid(true);

        let world_landmarks_out =
            get_type_and_quantization!(self.model_resource, self.world_landmarks_buf_index);
        let world_landmarks_shape = model_resource_check_and_get_impl!(
            self.model_resource,
            output_tensor_shape,
            self.world_landmarks_buf_index
        );
        let mut tensors_to_world_landmarks = TensorsToLandmarks::new(
            Self::RAW_LANDMARKS_NUM,
            world_landmarks_out,
            world_landmarks_shape,
        )?;
        tensors_to_world_landmarks
            .set_image_size(image_to_tensor_info.width(), image_to_tensor_info.height());

        let tensors_to_segmentation = if self.output_segmentation_masks() {
            let segmentation_shape = model_resource_check_and_get_impl!(
                self.model_resource,
                output_tensor_shape,
                self.segmentation_buf_index
            );
            Some(TensorsToSegmentation::new(
                crate::postprocess::Activation::SIGMOID,
                get_type_and_quantization!(self.model_resource, self.segmentation_buf_index),
                image_to_tensor_info.image_data_layout,
                segmentation_shape,
            )?)
        } else {
            None
        };

        let pose_detector_session = self.pose_detector.new_session()?;
        let execution_ctx = self.graph.init_execution_context()?;

        Ok(PoseLandmarkerSession {
            pose_landmarker: self,
            execution_ctx,
            pose_detector_session,
            image_to_tensor_info,
            input_tensor_shape,
            input_buffer: vec![0; tensor_bytes!(self.input_tensor_type, input_tensor_shape)],
            score_of_pose_presence: [0.],
            tensors_to_landmarks,
            tensors_to_world_landmarks,
            tensors_to_segmentation,
        })
    }
}

/// Session to run inference.
/// If process multiple images or videos, reuse it can get better performance.
pub struct PoseLandmarkerSession<'model> {
    pose_landmarker: &'model PoseLandmarker,
    execution_ctx: GraphExecutionContext<'model>,

    pose_detector_session: PoseDetectorSession<'model>,

    image_to_tensor_info: &'model ImageToTensorInfo,
    input_tensor_shape: &'model [usize],
    input_buffer: Vec<u8>,
    score_of_pose_presence: [f32; 1],
    tensors_to_landmarks: TensorsToLandmarks,
    tensors_to_world_landmarks: TensorsToLandmarks,
    tensors_to_segmentation: Option<TensorsToSegmentation>,
}

impl<'model> PoseLandmarkerSession<'model> {
    const DETECTION_TO_RECT_ROTATION_OPTION: Option<(f32, usize, usize)> =
        Some((90. * std::f32::consts::PI / 180.0, 0, 1));

    /// Detect one image using this task session.
    #[inline(always)]
    pub fn detect(&mut self, input: &impl ImageToTensor) -> Result<PoseLandmarkResults, Error> {
        let img_size = input.image_size();
        let pose_detection_result = self.pose_detector_session.detect(input)?;
        let mut pose_landmark_results = Vec::with_capacity(pose_detection_result.detections.len());

        for detection in pose_detection_result.detections.iter() {
            let pose_rect = NormalizedRect::from_detection(
                detection,
                Self::DETECTION_TO_RECT_ROTATION_OPTION,
                img_size.0,
                img_size.1,
                false,
            )
            .transform(img_size.0, img_size.1, 1.25, 1.25, 0.0, 0.0, None, true);

            input.to_tensor(
                self.image_to_tensor_info,
                &super::ImageProcessingOptions::from_normalized_rect(&pose_rect),
                &mut self.input_buffer,
            )?;

            self.execution_ctx.set_input(
                0,
                self.pose_landmarker.input_tensor_type,
                self.input_tensor_shape,
                self.input_buffer.as_slice(),
            )?;
            self.execution_ctx.compute()?;

            self.execution_ctx.get_output(
                self.pose_landmarker.score_buf_index,
                &mut self.score_of_pose_presence,
            )?;
            if self.score_of_pose_presence[0] < self.pose_landmarker.min_pose_presence_confidence()
            {
                continue;
            }

            self.execution_ctx.get_output(
                self.pose_landmarker.landmarks_buf_index,
                self.tensors_to_landmarks.landmark_buffer(),
            )?;
            let mut raw_pose_landmarks = self.tensors_to_landmarks.result(true).0;
            let mut pose_landmarks = Landmarks(
                raw_pose_landmarks
                    .drain(..PoseLandmark::NAMES.len())
                    .collect::<Vec<_>>(),
            );

            self.execution_ctx.get_output(
                self.pose_landmarker.world_landmarks_buf_index,
                self.tensors_to_world_landmarks.landmark_buffer(),
            )?;
            let mut raw_pose_world_landmarks = self.tensors_to_world_landmarks.result(false).0;
            let mut pose_world_landmarks = Landmarks(
                raw_pose_world_landmarks
                    .drain(..PoseLandmark::NAMES.len())
                    .collect::<Vec<_>>(),
            );

            for (world, normalized) in pose_world_landmarks.iter_mut().zip(pose_landmarks.iter()) {
                world.visibility = normalized.visibility;
                world.presence = normalized.presence;
            }

            projection_normalized_landmarks(&mut pose_landmarks, &pose_rect, false);
            projection_world_landmark(&mut pose_world_landmarks, &pose_rect);

            let segmentation_mask =
                if let Some(ref mut tensors_to_segmentation) = self.tensors_to_segmentation {
                self.execution_ctx.get_output(
                    self.pose_landmarker.segmentation_buf_index,
                    tensors_to_segmentation.tensor_buffer(),
                )?;
                    Some(resize_confidence_mask(
                        tensors_to_segmentation
                            .confidence_masks()
                            .into_iter()
                            .next()
                            .unwrap(),
                        img_size,
                    ))
                } else {
                    None
                };

            pose_landmark_results.push(PoseLandmarkResult {
                pose_landmarks,
                pose_world_landmarks,
                segmentation_mask,
            });
        }

        Ok(PoseLandmarkResults(pose_landmark_results))
    }

    /// Detect input video stream use this session.
    /// Return a iterator for results, process input stream when poll next result.
    #[inline(always)]
    pub fn detect_for_video<InputVideoData: VideoData>(
        &mut self,
        video_data: InputVideoData,
    ) -> Result<VideoResultsIter<Self, InputVideoData>, Error> {
        Ok(VideoResultsIter::new(self, video_data))
    }
}

impl<'model> super::TaskSession for PoseLandmarkerSession<'model> {
    type Result = PoseLandmarkResults;

    #[inline]
    fn process_next(
        &mut self,
        _process_options: &super::ImageProcessingOptions,
        video_data: &mut impl VideoData,
    ) -> Result<Option<Self::Result>, Error> {
        if let Some(frame) = video_data.next_frame()? {
            return self.detect(&frame).map(Some);
        }
        Ok(None)
    }
}

struct PoseDetector {
    model_resource: Box<dyn ModelResourceTrait>,
    graph: Graph,
    anchors: Vec<Anchor>,
    min_detection_confidence: f32,
    num_poses: i32,
    input_tensor_type: TensorType,
}

impl PoseDetector {
    const POSE_LABELS: &'static [u8] = b"Pose";
    const NUM_BOXES: usize = 2254;

    fn new(
        device: Device,
        num_poses: i32,
        min_detection_confidence: f32,
        buffer: impl AsRef<[u8]>,
    ) -> Result<Self, Error> {
        let buf = buffer.as_ref();
        let model_resource = crate::model::parse_model(buf)?;
        model_base_check_impl!(model_resource, 1, 2);
        let img_info =
            model_resource_check_and_get_impl!(model_resource, to_tensor_info, 0).try_to_image()?;
        let anchors = crate::postprocess::SsdAnchorsBuilder::new(
            img_info.width(),
            img_info.height(),
            0.1484375,
            0.75,
            5,
        )
        .anchor_offset_x(0.5)
        .anchor_offset_y(0.5)
        .strides(vec![8, 16, 32, 32, 32])
        .aspect_ratios(vec![1.0])
        .fixed_anchor_size(true)
        .generate();

        let graph = crate::GraphBuilder::new(model_resource.model_backend(), device)
            .build_from_bytes([buf])?;
        let input_tensor_type =
            model_resource_check_and_get_impl!(model_resource, input_tensor_type, 0);

        Ok(Self {
            model_resource,
            graph,
            anchors,
            min_detection_confidence,
            num_poses,
            input_tensor_type,
        })
    }

    fn new_session(&self) -> Result<PoseDetectorSession, Error> {
        let image_to_tensor_info =
            model_resource_check_and_get_impl!(self.model_resource, to_tensor_info, 0)
                .try_to_image()?;
        let input_tensor_shape =
            model_resource_check_and_get_impl!(self.model_resource, input_tensor_shape, 0);
        let categories_filter =
            CategoriesFilter::new_full(self.min_detection_confidence, Self::POSE_LABELS, None);
        let mut tensors_to_detection = TensorsToDetection::new_with_anchors(
            categories_filter,
            &self.anchors,
            self.min_detection_confidence,
            self.num_poses,
            get_type_and_quantization!(self.model_resource, 0),
            get_type_and_quantization!(self.model_resource, 1),
        );
        tensors_to_detection.set_anchors_scales(224.0, 224.0, 224.0, 224.0);
        tensors_to_detection.set_num_coords(12);
        tensors_to_detection.set_key_points(4, 2, 4);
        tensors_to_detection.set_sigmoid_score(true);
        tensors_to_detection.set_score_clipping_thresh(100.);
        tensors_to_detection.set_box_format(DetectionBoxFormat::YXHW);
        tensors_to_detection.set_nms_min_suppression_threshold(0.3);
        tensors_to_detection
            .set_nms_overlap_type(NonMaxSuppressionOverlapType::IntersectionOverUnion);
        tensors_to_detection.set_nms_algorithm(NonMaxSuppressionAlgorithm::WEIGHTED);
        tensors_to_detection.realloc(Self::NUM_BOXES);

        let execution_ctx = self.graph.init_execution_context()?;
        Ok(PoseDetectorSession {
            detector: self,
            execution_ctx,
            tensors_to_detection,
            image_to_tensor_info,
            input_tensor_shape,
            input_buffer: vec![0; tensor_bytes!(self.input_tensor_type, input_tensor_shape)],
        })
    }
}

struct PoseDetectorSession<'model> {
    detector: &'model PoseDetector,
    execution_ctx: GraphExecutionContext<'model>,
    tensors_to_detection: TensorsToDetection<'model>,
    image_to_tensor_info: &'model ImageToTensorInfo,
    input_tensor_shape: &'model [usize],
    input_buffer: Vec<u8>,
}

impl<'model> PoseDetectorSession<'model> {
    #[inline(always)]
    fn compute(&mut self) -> Result<DetectionResult, Error> {
        self.execution_ctx.set_input(
            0,
            self.detector.input_tensor_type,
            self.input_tensor_shape,
            self.input_buffer.as_slice(),
        )?;
        self.execution_ctx.compute()?;
        self.execution_ctx
            .get_output(0, self.tensors_to_detection.location_buf())?;
        self.execution_ctx
            .get_output(1, self.tensors_to_detection.score_buf())?;
        Ok(self.tensors_to_detection.result(PoseDetector::NUM_BOXES))
    }

    #[inline(always)]
    fn detect(&mut self, input: &impl ImageToTensor) -> Result<DetectionResult, Error> {
        input.to_tensor(
            self.image_to_tensor_info,
            &Default::default(),
            &mut self.input_buffer,
        )?;
        self.compute()
    }
}

fn resize_confidence_mask(mask: ImageConfidenceMask, img_size: (u32, u32)) -> ImageConfidenceMask {
    if mask.dimensions() == img_size {
        mask
    } else {
        image::imageops::resize(
            &mask,
            img_size.0,
            img_size.1,
            image::imageops::FilterType::Triangle,
        )
    }
}
