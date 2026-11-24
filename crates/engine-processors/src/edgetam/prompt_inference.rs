//! Deterministic prompt/candidate/mask policy shared by still-image and video
//! sessions. Inference execution remains in [`super::model`].
//!
//! Ported from `Processors/EdgeTAM/EdgeTAMPromptInference.swift`.

use ferment_engine_core::{VideoObjectMask, VideoObjectPrompt};
use half::f16;

use super::error::EdgeTamError;
use super::model::EdgeTamDecodeOutput;
use super::processing::{
    self, BilinearResizePlan, LOW_RESOLUTION_MASK_SIZE, ProcessingError, PromptTensor,
};

pub const MAXIMUM_PROMPT_TOKENS: usize = 8;

/// Session-owned display-mask scratch. Source dimensions are immutable for a
/// session, so both the interpolation coordinates and the promoted 256×256
/// logits stay fixed-size across arbitrarily long videos.
#[derive(Debug)]
pub struct MaskWorkspace {
    plan: BilinearResizePlan,
    promoted_logits: Vec<f32>,
}

impl MaskWorkspace {
    pub fn new(source_width: usize, source_height: usize) -> Result<Self, ProcessingError> {
        Ok(Self {
            plan: processing::display_mask_plan(source_width, source_height)?,
            promoted_logits: vec![0.0_f32; LOW_RESOLUTION_MASK_SIZE * LOW_RESOLUTION_MASK_SIZE],
        })
    }

    pub fn mask(&mut self, values: &[f16]) -> Result<VideoObjectMask, ProcessingError> {
        if values.len() != self.promoted_logits.len() {
            return Err(ProcessingError::InvalidElementCount {
                expected: self.promoted_logits.len(),
                actual: values.len(),
            });
        }
        for (destination, source) in self.promoted_logits.iter_mut().zip(values.iter()) {
            *destination = source.to_f32();
        }
        processing::thresholded_display_mask_with_plan(&self.promoted_logits, &self.plan)
    }

    #[must_use]
    pub fn retained_byte_count(&self) -> usize {
        self.plan.retained_byte_count() + self.promoted_logits.len() * std::mem::size_of::<f32>()
    }
}

pub fn prompt_tensor(
    prompt: &VideoObjectPrompt,
    replacing_existing_prompts: bool,
    previous: Option<&PromptTensor>,
) -> Result<PromptTensor, EdgeTamError> {
    let new = processing::prompt_tensor(prompt)
        .map_err(|error| EdgeTamError::InvalidModelBoundary(error.to_string()))?;

    let result = if replacing_existing_prompts || previous.is_none() {
        new
    } else {
        match prompt {
            VideoObjectPrompt::Points(_) => {}
            VideoObjectPrompt::Box(_) | VideoObjectPrompt::BoxAndPoints { .. } => {
                return Err(EdgeTamError::CannotAppendBox);
            }
        }
        let Some(previous) = previous else {
            return Err(EdgeTamError::InvalidModelBoundary(
                "missing previous prompt after append validation".into(),
            ));
        };
        PromptTensor {
            coordinates: [previous.coordinates.as_slice(), new.coordinates.as_slice()].concat(),
            labels: [previous.labels.as_slice(), new.labels.as_slice()].concat(),
        }
    };

    if result.point_count() > MAXIMUM_PROMPT_TOKENS {
        return Err(EdgeTamError::TooManyPromptTokens {
            actual: result.point_count(),
            maximum: MAXIMUM_PROMPT_TOKENS,
        });
    }
    Ok(result)
}

/// The previous-mask decoder input: the selected candidate's logits clamped to
/// `[-32, 32]` in `f32` and rounded back to `f16`.
#[must_use]
pub fn previous_mask_input(values: &[f16]) -> Vec<f16> {
    values
        .iter()
        .map(|value| f16::from_f32(value.to_f32().clamp(-32.0, 32.0)))
        .collect()
}

/// Stock policy: candidate zero for boxes/multiple clicks; for one-click and
/// tracking prompts, argmax over candidates one through three with
/// first-on-tie semantics.
pub fn selected_candidate_index(
    output: &EdgeTamDecodeOutput,
    uses_multimask: bool,
) -> Result<usize, EdgeTamError> {
    if !uses_multimask {
        return Ok(0);
    }
    let mut best_index = 1;
    let mut best_iou = output.iou_prediction(best_index)?;
    for index in 2..=3 {
        let iou = output.iou_prediction(index)?;
        // torch.argmax keeps the first candidate on an exact tie.
        if iou > best_iou {
            best_index = index;
            best_iou = iou;
        }
    }
    Ok(best_index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferment_engine_core::{
        NormalizedVideoBox, NormalizedVideoPoint, VideoObjectPointLabel, VideoObjectPointPrompt,
    };

    fn point(x: f32, label: VideoObjectPointLabel) -> VideoObjectPointPrompt {
        VideoObjectPointPrompt::new(NormalizedVideoPoint::new(x, 0.5), label)
    }

    /// Appending a click adds a TOKEN; replacing throws the old ones away.
    /// This is the whole difference between an editor's second tap correcting a
    /// selection and starting a new one, and it lives in the tensor — the masks
    /// a model returns for the two cannot be told apart on a synthetic fixture.
    ///
    /// kill: ignore `replacing_existing_prompts` and always take `new` — the
    /// appended tensor drops to one token and the first click is forgotten.
    #[test]
    fn appending_keeps_the_earlier_clicks_and_replacing_drops_them() {
        let first = prompt_tensor(
            &VideoObjectPrompt::Points(vec![point(0.25, VideoObjectPointLabel::Foreground)]),
            true,
            None,
        )
        .expect("the first prompt");
        assert_eq!(first.labels, [1]);
        assert_eq!(first.coordinates.len(), 2);

        let appended = prompt_tensor(
            &VideoObjectPrompt::Points(vec![point(0.75, VideoObjectPointLabel::Background)]),
            false,
            Some(&first),
        )
        .expect("the appended prompt");
        assert_eq!(appended.labels, [1, 0], "the append must keep BOTH clicks");
        assert_eq!(appended.coordinates.len(), 4);
        assert_eq!(&appended.coordinates[..2], &first.coordinates[..]);

        let replaced = prompt_tensor(
            &VideoObjectPrompt::Points(vec![point(0.75, VideoObjectPointLabel::Background)]),
            true,
            Some(&first),
        )
        .expect("the replacing prompt");
        assert_eq!(
            replaced.labels,
            [0],
            "replacing must drop the earlier click"
        );
    }

    /// A BOX can only start or replace a prompt: EdgeTAM needs its two corner
    /// tokens first, and an append would put them after existing points.
    ///
    /// kill: let the `box`/`boxAndPoints` arms fall through to the append and
    /// the corner labels land in the middle of the token list, where the prompt
    /// encoder reads them as ordinary clicks.
    #[test]
    fn a_box_cannot_be_appended() {
        let first = prompt_tensor(
            &VideoObjectPrompt::Points(vec![point(0.25, VideoObjectPointLabel::Foreground)]),
            true,
            None,
        )
        .expect("the first prompt");
        let bounds = NormalizedVideoBox::from_extents(0.1, 0.1, 0.9, 0.9);

        assert!(matches!(
            prompt_tensor(&VideoObjectPrompt::Box(bounds), false, Some(&first)),
            Err(EdgeTamError::CannotAppendBox)
        ));
        assert!(matches!(
            prompt_tensor(
                &VideoObjectPrompt::BoxAndPoints {
                    r#box: bounds,
                    points: vec![point(0.5, VideoObjectPointLabel::Foreground)],
                },
                false,
                Some(&first),
            ),
            Err(EdgeTamError::CannotAppendBox)
        ));
        // Replacing with a box is fine, and the corners come FIRST.
        let replaced = prompt_tensor(&VideoObjectPrompt::Box(bounds), true, Some(&first))
            .expect("a box may replace");
        assert_eq!(replaced.labels, [2, 3]);
    }

    /// Nine tokens is more than this build's decoder has: the ninth click is
    /// refused with the count, not silently dropped or sent to a graph whose
    /// token axis is eight.
    ///
    /// kill: raise `MAXIMUM_PROMPT_TOKENS` past the exported graph's static
    /// axis and the ninth click reaches ORT, which rejects the shape with a
    /// message about dimensions rather than about clicks.
    #[test]
    fn the_ninth_click_is_refused_with_its_count() {
        let mut tensor = prompt_tensor(
            &VideoObjectPrompt::Points(vec![point(0.1, VideoObjectPointLabel::Foreground)]),
            true,
            None,
        )
        .expect("the first prompt");
        for index in 1..8 {
            tensor = prompt_tensor(
                &VideoObjectPrompt::Points(vec![point(
                    0.1 * index as f32,
                    VideoObjectPointLabel::Foreground,
                )]),
                false,
                Some(&tensor),
            )
            .unwrap_or_else(|error| panic!("click {index}: {error}"));
        }
        assert_eq!(tensor.point_count(), MAXIMUM_PROMPT_TOKENS);

        let ninth = prompt_tensor(
            &VideoObjectPrompt::Points(vec![point(0.9, VideoObjectPointLabel::Foreground)]),
            false,
            Some(&tensor),
        );
        assert!(matches!(
            ninth,
            Err(EdgeTamError::TooManyPromptTokens {
                actual: 9,
                maximum: 8
            })
        ));
    }

    /// The previous mask crosses back into the decoder CLAMPED to ±32.
    ///
    /// Raw decoder logits reach ±1e4 on a confident mask; feeding those back as
    /// a dense prompt saturates the mask encoder. iOS clamps the candidate's
    /// own array in place, this port builds the clamped copy — the bytes the
    /// graph sees are the same either way, which is what this pins.
    ///
    /// kill: drop the clamp and the extremes below travel unchanged.
    #[test]
    fn the_previous_mask_is_clamped_to_the_decoders_range() {
        let raw: Vec<half::f16> = [-1.0e4_f32, -33.0, -32.0, 0.0, 31.5, 32.0, 1.0e4]
            .into_iter()
            .map(half::f16::from_f32)
            .collect();
        let clamped = previous_mask_input(&raw);
        assert_eq!(
            clamped.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            [-32.0, -32.0, -32.0, 0.0, 31.5, 32.0, 32.0]
        );
    }
}
