//! The greedy TDT decode of one window — the contract's `decode`, which is
//! `ParakeetForTDT.generate`'s: the decoder once on blank from zero states;
//! each step the joint over the frame and the decoder's output, the token the
//! argmax of the vocabulary span and the duration the argmax of the duration
//! span; a blank's duration 0 is 1; a non-blank token is emitted at its frame
//! and the decoder runs on it (a blank leaves the decoder as it is); the frame
//! advances by the duration until the window's valid frames, in at most
//! `MAX_STEPS` steps. The decoder and the joint are the host's
//! ([`HostDecode`]).

use ferment_engine_core::{EngineError, Result};

use super::config::{
    BLANK_TOKEN_ID, DURATION_BINS, ENCODER_FRAMES, HIDDEN, LOGITS, MAX_STEPS, VOCAB,
};
use super::host_decode::{HostDecode, Prediction};

/// A window's encoder output, frame-major `[frames][640]`, and how many of
/// its frames are real.
pub struct Encoded {
    frames: Vec<f32>,
    valid: usize,
}

impl Encoded {
    /// The door's `encoded [1, 640, 1, 188]` (channels major), `valid` of its
    /// frames real.
    pub fn from_door(values: &[f32], valid: usize) -> Result<Self> {
        if values.len() != HIDDEN * ENCODER_FRAMES {
            return Err(EngineError::inference(format!(
                "parakeet_encoder answered {} values, the door is {HIDDEN} × {ENCODER_FRAMES}",
                values.len()
            )));
        }
        let mut frames = vec![0.0; values.len()];
        for channel in 0..HIDDEN {
            for frame in 0..ENCODER_FRAMES {
                frames[frame * HIDDEN + channel] = values[channel * ENCODER_FRAMES + frame];
            }
        }
        Ok(Self {
            frames,
            valid: valid.min(ENCODER_FRAMES),
        })
    }

    #[must_use]
    pub fn valid(&self) -> usize {
        self.valid
    }

    #[must_use]
    pub fn frame(&self, index: usize) -> &[f32] {
        &self.frames[index * HIDDEN..(index + 1) * HIDDEN]
    }
}

/// What a window emits: each token, the encoder frame it came at, its
/// probability and its predicted duration.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WindowTokens {
    pub tokens: Vec<i32>,
    pub frames: Vec<usize>,
    pub confidences: Vec<f32>,
    pub durations: Vec<i32>,
}

/// One joint answer: the two argmaxes, and the token's probability — the
/// softmax of the vocabulary span at its argmax.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decision {
    pub token: usize,
    pub duration: usize,
    pub probability: f32,
}

impl Decision {
    /// `logits [8198]`: the vocabulary, then the five durations. The first
    /// maximum wins a tie, as `argmax` does.
    #[must_use]
    pub fn from_logits(logits: &[f32]) -> Self {
        let first_max = |span: &[f32]| {
            span.iter()
                .enumerate()
                .fold((0, f32::NEG_INFINITY), |best, (index, &value)| {
                    if value > best.1 { (index, value) } else { best }
                })
        };
        let (token, peak) = first_max(&logits[..VOCAB]);
        let sum: f32 = logits[..VOCAB]
            .iter()
            .map(|value| (value - peak).exp())
            .sum();
        let (bin, _) = first_max(&logits[VOCAB..LOGITS]);
        Self {
            token,
            duration: DURATION_BINS[bin],
            probability: 1.0 / sum,
        }
    }
}

pub fn decode_window(weights: &HostDecode, encoded: &Encoded) -> WindowTokens {
    let mut prediction = weights.predict(BLANK_TOKEN_ID, &Prediction::zero());
    let mut logits = vec![0.0; LOGITS];
    let mut emitted = WindowTokens::default();
    let (mut frame, mut steps) = (0, 0);
    while frame < encoded.valid && steps < MAX_STEPS {
        weights.joint(encoded.frame(frame), &prediction.output, &mut logits);
        let decision = Decision::from_logits(&logits);
        let mut duration = decision.duration;
        if decision.token == BLANK_TOKEN_ID {
            duration = duration.max(1);
        } else {
            emitted.tokens.push(decision.token as i32);
            emitted.frames.push(frame);
            emitted.confidences.push(decision.probability);
            emitted.durations.push(decision.duration as i32);
            prediction = weights.predict(decision.token, &prediction);
        }
        frame += duration;
        steps += 1;
    }
    emitted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_decision_is_the_two_argmaxes_and_the_tokens_softmax() {
        let mut logits = vec![0.0; LOGITS];
        logits[42] = 3.0;
        logits[VOCAB + 2] = 1.0;
        let decision = Decision::from_logits(&logits);
        assert_eq!(decision.token, 42);
        assert_eq!(decision.duration, 2);
        let expected = 3.0_f32.exp() / (3.0_f32.exp() + (VOCAB - 1) as f32);
        assert!((decision.probability - expected).abs() < 1e-6);
    }

    #[test]
    fn a_tie_goes_to_the_first_index() {
        let mut logits = vec![0.0; LOGITS];
        logits[7] = 5.0;
        logits[9] = 5.0;
        assert_eq!(Decision::from_logits(&logits).token, 7);
        assert_eq!(Decision::from_logits(&logits).duration, 0);
    }

    #[test]
    fn the_doors_encoded_is_read_channels_major() {
        let values: Vec<f32> = (0..HIDDEN * ENCODER_FRAMES).map(|i| i as f32).collect();
        let encoded = Encoded::from_door(&values, 500).unwrap();
        assert_eq!(encoded.valid(), ENCODER_FRAMES);
        assert_eq!(encoded.frame(3)[0], 3.0);
        assert_eq!(encoded.frame(3)[1], (ENCODER_FRAMES + 3) as f32);
        assert!(Encoded::from_door(&values[1..], 10).is_err());
    }
}
