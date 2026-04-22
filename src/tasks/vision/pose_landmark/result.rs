use super::PoseLandmark;
use crate::postprocess::utils::{draw_landmarks_with_options, DefaultPixel, DrawLandmarksOptions};
use crate::postprocess::{ImageConfidenceMask, Landmarks, NormalizedLandmarks};
use std::fmt::{Display, Formatter};
use std::ops::{Deref, DerefMut};

/// A single pose landmark detection result.
#[derive(Debug)]
pub struct PoseLandmarkResult {
    /// Detected pose landmarks in normalized image coordinates.
    pub pose_landmarks: NormalizedLandmarks,
    /// Detected pose landmarks in world coordinates.
    pub pose_world_landmarks: Landmarks,
    /// Optional pose segmentation mask in confidence image format.
    pub segmentation_mask: Option<ImageConfidenceMask>,
}

impl PoseLandmarkResult {
    /// Draw this detection result to image with default options
    #[inline(always)]
    pub fn draw<I>(&self, img: &mut I)
    where
        I: image::GenericImage,
        I::Pixel: 'static + DefaultPixel,
        <I::Pixel as image::Pixel>::Subpixel: Into<f32> + imageproc::definitions::Clamp<f32>,
    {
        let mut options = DrawLandmarksOptions::default();
        options.connections = PoseLandmark::CONNECTIONS;
        draw_landmarks_with_options(img, &self.pose_landmarks, &options);
    }

    /// Draw this detection result to image with options
    #[inline(always)]
    pub fn draw_with_options<I>(&self, img: &mut I, options: &DrawLandmarksOptions<I::Pixel>)
    where
        I: image::GenericImage,
        I::Pixel: 'static,
        <I::Pixel as image::Pixel>::Subpixel: Into<f32> + imageproc::definitions::Clamp<f32>,
    {
        draw_landmarks_with_options(img, &self.pose_landmarks, options);
    }
}

/// The pose landmarks detection result from PoseLandmarker.
#[derive(Debug)]
pub struct PoseLandmarkResults(pub Vec<PoseLandmarkResult>);

impl Deref for PoseLandmarkResults {
    type Target = Vec<PoseLandmarkResult>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for PoseLandmarkResults {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl IntoIterator for PoseLandmarkResults {
    type Item = PoseLandmarkResult;
    type IntoIter = std::vec::IntoIter<PoseLandmarkResult>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Display for PoseLandmarkResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  Landmarks:")?;
        for (i, l) in self.pose_landmarks.iter().enumerate() {
            writeln!(
                f,
                "    Normalized Landmark #{} ({}):",
                i,
                PoseLandmark::NAMES[i]
            )?;
            write!(f, "{}", l)?;
        }
        writeln!(f, "  WorldLandmarks:")?;
        for (i, l) in self.pose_world_landmarks.iter().enumerate() {
            writeln!(f, "    Landmark #{} ({}):", i, PoseLandmark::NAMES[i])?;
            write!(f, "{}", l)?;
        }
        if self.segmentation_mask.is_some() {
            writeln!(f, "  SegmentationMask: Some")?;
        }
        Ok(())
    }
}

impl Display for PoseLandmarkResults {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            writeln!(f, "No PoseLandmarkResult.")?;
        } else {
            for (i, r) in self.iter().enumerate() {
                writeln!(f, "PoseLandmarkResult #{}", i)?;
                write!(f, "{}", r)?;
            }
        }
        Ok(())
    }
}
