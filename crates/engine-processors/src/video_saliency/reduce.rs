//! Pure: samples -> one verdict per scene, in scene order. Port of Swift
//! `VideoSaliencyReduce`.
//!
//! Medians, not means: a single grab that landed on a cutaway or a false face
//! must not drag the scene's verdict across the frame.

use super::{SampleSource, VideoSaliencySample, VideoSaliencySceneVerdict};

impl VideoSaliencySceneVerdict {
    /// Every grab in the scene failed: centred, full-frame, no signal.
    pub(crate) fn no_samples(index: usize) -> Self {
        Self {
            index,
            sample_count: 0,
            cx: 0.5,
            cy: 0.5,
            w: 1.0,
            h: 1.0,
            confidence: 0.0,
            face_share: 0.0,
            spread: 0.0,
        }
    }
}

pub fn verdicts(
    samples: &[VideoSaliencySample],
    scene_count: usize,
) -> Vec<VideoSaliencySceneVerdict> {
    let mut grouped: Vec<Vec<&VideoSaliencySample>> = vec![Vec::new(); scene_count];
    for sample in samples {
        if let Some(group) = grouped.get_mut(sample.scene_index) {
            group.push(sample);
        }
    }

    grouped
        .into_iter()
        .enumerate()
        .map(|(index, group)| {
            if group.is_empty() {
                return VideoSaliencySceneVerdict::no_samples(index);
            }
            let cx = median(group.iter().map(|s| s.cx));
            let cy = median(group.iter().map(|s| s.cy));
            let faces = group
                .iter()
                .filter(|s| s.source == SampleSource::Face)
                .count();
            let squares: f64 = group
                .iter()
                .map(|s| (cx - s.cx).powi(2) + (cy - s.cy).powi(2))
                .sum();
            VideoSaliencySceneVerdict {
                index,
                sample_count: group.len(),
                cx,
                cy,
                w: median(group.iter().map(|s| s.w)),
                h: median(group.iter().map(|s| s.h)),
                confidence: median(group.iter().map(|s| s.confidence)),
                face_share: faces as f64 / group.len() as f64,
                spread: (squares / group.len() as f64).sqrt(),
            }
        })
        .collect()
}

/// Lower median — an even count takes the smaller of the two middles.
/// Panics on an empty input, which [`verdicts`] has already excluded.
pub fn median(values: impl Iterator<Item = f64>) -> f64 {
    let mut sorted: Vec<f64> = values.collect();
    sorted.sort_by(f64::total_cmp);
    sorted[(sorted.len() - 1) / 2]
}

#[cfg(test)]
mod tests {
    //! Ported from `VideoSaliencyReduceTests.swift`. Hand-built sample sets —
    //! no video, no model. Pins the median convention (lower median on even
    //! counts), the spread definition, and the shape contract that every input
    //! scene gets exactly one verdict.

    use super::*;

    #[derive(Clone, Copy)]
    struct Spec {
        scene: usize,
        time: f64,
        cx: f64,
        cy: f64,
        w: f64,
        h: f64,
        confidence: f64,
        source: SampleSource,
    }

    impl Default for Spec {
        fn default() -> Self {
            Self {
                scene: 0,
                time: 0.0,
                cx: 0.5,
                cy: 0.5,
                w: 0.2,
                h: 0.2,
                confidence: 0.5,
                source: SampleSource::Saliency,
            }
        }
    }

    fn sample(spec: Spec) -> VideoSaliencySample {
        VideoSaliencySample {
            scene_index: spec.scene,
            time: spec.time,
            cx: spec.cx,
            cy: spec.cy,
            w: spec.w,
            h: spec.h,
            confidence: spec.confidence,
            source: spec.source,
        }
    }

    #[test]
    fn median_of_an_odd_count_is_the_middle_value() {
        let verdicts = verdicts(
            &[
                sample(Spec {
                    time: 0.0,
                    cx: 0.8,
                    w: 0.5,
                    confidence: 0.9,
                    ..Spec::default()
                }),
                sample(Spec {
                    time: 1.0,
                    cx: 0.2,
                    w: 0.1,
                    confidence: 0.1,
                    ..Spec::default()
                }),
                sample(Spec {
                    time: 2.0,
                    cx: 0.5,
                    w: 0.3,
                    confidence: 0.4,
                    ..Spec::default()
                }),
            ],
            1,
        );
        assert_eq!(verdicts.len(), 1);
        let verdict = verdicts[0];
        assert!((verdict.cx - 0.5).abs() < 1e-12);
        assert!((verdict.w - 0.3).abs() < 1e-12);
        assert!((verdict.confidence - 0.4).abs() < 1e-12);
        assert_eq!(verdict.sample_count, 3);
    }

    #[test]
    fn even_count_takes_the_lower_median() {
        let verdicts = verdicts(
            &[0.4, 0.1, 0.3, 0.2]
                .iter()
                .enumerate()
                .map(|(index, &cx)| {
                    sample(Spec {
                        time: index as f64,
                        cx,
                        ..Spec::default()
                    })
                })
                .collect::<Vec<_>>(),
            1,
        );
        assert!((verdicts[0].cx - 0.2).abs() < 1e-12);
    }

    /// Centroids at x = 0.2 / 0.5 / 0.8 (y fixed): median 0.5, distances
    /// 0.3 / 0 / 0.3 -> RMS = sqrt(0.18 / 3) = 0.2449.
    #[test]
    fn spread_is_the_rms_distance_from_the_median_centroid() {
        let verdicts = verdicts(
            &[0.2, 0.5, 0.8]
                .iter()
                .enumerate()
                .map(|(index, &cx)| {
                    sample(Spec {
                        time: index as f64,
                        cx,
                        cy: 0.5,
                        ..Spec::default()
                    })
                })
                .collect::<Vec<_>>(),
            1,
        );
        assert!((verdicts[0].spread - 0.244_948_974).abs() < 1e-6);
    }

    #[test]
    fn a_still_subject_has_zero_spread() {
        let verdicts = verdicts(
            &(0..4)
                .map(|index| {
                    sample(Spec {
                        time: f64::from(index),
                        cx: 0.31,
                        cy: 0.62,
                        ..Spec::default()
                    })
                })
                .collect::<Vec<_>>(),
            1,
        );
        assert!(verdicts[0].spread.abs() < 1e-12);
        assert!((verdicts[0].cx - 0.31).abs() < 1e-12);
        assert!((verdicts[0].cy - 0.62).abs() < 1e-12);
    }

    #[test]
    fn face_share_is_the_fraction_of_face_samples() {
        let sources = [
            SampleSource::Face,
            SampleSource::Face,
            SampleSource::Face,
            SampleSource::Saliency,
        ];
        let verdicts = verdicts(
            &sources
                .iter()
                .enumerate()
                .map(|(index, &source)| {
                    sample(Spec {
                        time: index as f64,
                        source,
                        ..Spec::default()
                    })
                })
                .collect::<Vec<_>>(),
            1,
        );
        assert!((verdicts[0].face_share - 0.75).abs() < 1e-12);
    }

    #[test]
    fn a_scene_with_no_samples_is_the_centred_full_frame_default() {
        let verdicts = verdicts(&[], 1);
        assert_eq!(verdicts, vec![VideoSaliencySceneVerdict::no_samples(0)]);
    }

    #[test]
    fn every_input_scene_gets_a_verdict_even_when_a_middle_one_is_empty() {
        let verdicts = verdicts(
            &[
                sample(Spec {
                    scene: 0,
                    time: 0.0,
                    cx: 0.25,
                    ..Spec::default()
                }),
                sample(Spec {
                    scene: 2,
                    time: 9.0,
                    cx: 0.75,
                    ..Spec::default()
                }),
            ],
            3,
        );
        assert_eq!(
            verdicts.iter().map(|v| v.index).collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert_eq!(
            verdicts.iter().map(|v| v.sample_count).collect::<Vec<_>>(),
            [1, 0, 1]
        );
        assert_eq!(
            verdicts.iter().map(|v| v.cx).collect::<Vec<_>>(),
            [0.25, 0.5, 0.75]
        );
    }
}
