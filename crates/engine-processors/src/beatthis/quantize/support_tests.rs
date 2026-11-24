use super::*;

fn meter() -> Option<MeterResult> {
    Some(MeterResult {
        meter: 4,
        raw_meter: 4,
        phase: 0,
    })
}

#[test]
fn real_dbn_breakdowns_do_not_create_four_spurious_tempo_excursions() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../testdata/dbn-weak-breakdowns.json")).unwrap();
    let beats: Vec<f64> = serde_json::from_value(fixture["beats"].clone()).unwrap();
    let down: Vec<f64> = serde_json::from_value(fixture["downbeats"].clone()).unwrap();
    let support: Vec<bool> = serde_json::from_value(fixture["support"].clone()).unwrap();
    let original = quantize_beats(&beats, &down, meter(), Some(517.9067));
    assert_eq!(
        original.segments.len(),
        9,
        "pins the actual reported failure"
    );
    let q = quantize_supported_beats(&beats, &down, meter(), Some(517.9067), &support);
    assert_eq!(q.segments.len(), 1, "{:?}", q.segments);
    assert!((q.bpm.unwrap() - 121.).abs() < 0.03, "{:?}", q.bpm);
    assert_eq!(q.beats.len(), 1028);
    let segment = &q.segments[0];
    let period = 60. / segment.bpm;
    let mut residual = beats
        .iter()
        .zip(&support)
        .filter(|(_, s)| **s)
        .map(|(t, _)| ((t - segment.first + period / 2.).rem_euclid(period) - period / 2.).abs())
        .collect::<Vec<_>>();
    residual.sort_by(f64::total_cmp);
    assert!(
        residual[residual.len() * 9 / 10] < 0.035,
        "p90={}",
        residual[residual.len() * 9 / 10]
    );
}

#[test]
fn inferred_breakdown_beats_do_not_become_tempo_changes_or_shrink_the_song() {
    let mut beats = (0..128).map(|i| 0.2 + i as f64 * 0.5).collect::<Vec<_>>();
    let down = beats.iter().step_by(4).copied().collect::<Vec<_>>();
    let mut support = vec![true; beats.len()];
    // A weak decoded path wanders for a bar, then returns to measured beats.
    for i in 32..39 {
        beats[i] += (i - 31) as f64 * 0.035;
        support[i] = false;
    }
    let q = quantize_supported_beats(&beats, &down, meter(), Some(64.), &support);
    assert_eq!(q.segments.len(), 1, "{:?}", q.segments);
    assert!((q.bpm.unwrap() - 120.).abs() < 1e-8);
    assert_eq!(
        q.beats.len(),
        128,
        "missing observations are bridges, not deleted time"
    );
    for (i, &beat) in q.beats.iter().enumerate() {
        assert!((beat - (0.2 + i as f64 * 0.5)).abs() < 1e-8);
    }
}

#[test]
fn confident_real_tempo_changes_are_exactly_the_legacy_segmented_result() {
    for middle_count in [8, 32, 128] {
        let beats = (0..64)
            .map(|i| 0.2 + i as f64 * 0.5)
            .chain((0..middle_count).map(|i| 32.2 + i as f64 * 0.4))
            .chain((0..64).map(|i| 32.2 + middle_count as f64 * 0.4 + i as f64 * 0.5))
            .collect::<Vec<_>>();
        let down = beats.iter().step_by(4).copied().collect::<Vec<_>>();
        let original = quantize_beats(&beats, &down, meter(), None);
        let supported =
            quantize_supported_beats(&beats, &down, meter(), None, &vec![true; beats.len()]);
        assert_eq!(supported.segments, original.segments);
        assert_eq!(supported.beats, original.beats);
        assert_eq!(supported.downbeats, original.downbeats);
        // The existing 17-IBI robust window treats an eight-beat excursion as
        // a fill. Evidence filtering must not change that established policy;
        // sustained tempo sections must still survive as separate runs.
        if middle_count >= 16 {
            assert_eq!(supported.segments.len(), 3);
            assert!(
                supported
                    .segments
                    .iter()
                    .any(|s| (s.bpm - 150.).abs() < 0.1)
            );
            assert!(
                supported
                    .segments
                    .iter()
                    .any(|s| (s.bpm - 120.).abs() < 0.1)
            );
        } else {
            // The single fitted slope is biased by the short excursion; it
            // is not a correct 120 BPM grid. Pin this existing limitation,
            // rather than presenting legacy parity as musical correctness.
            assert_eq!(original.segments.len(), 1);
            assert!((original.segments[0].bpm - 122.15319704422033).abs() < 1e-8);
        }
    }
}

#[test]
fn weak_evidence_never_invents_a_grid_and_unsupported_edges_are_not_extrapolated() {
    let beats = (0..40).map(|i| i as f64 * 0.5).collect::<Vec<_>>();
    assert!(
        quantize_supported_beats(&beats, &[], meter(), None, &[false; 40])
            .segments
            .is_empty()
    );
    assert!(
        quantize_supported_beats(&beats, &[], meter(), None, &[])
            .segments
            .is_empty()
    );
    let support = (0..40).map(|i| (8..32).contains(&i)).collect::<Vec<_>>();
    let q = quantize_supported_beats(&beats, &[], meter(), None, &support);
    assert_eq!(q.beats.len(), 24);
    assert!((q.beats[0] - 4.).abs() < 1e-10);
    assert!((q.beats[23] - 15.5).abs() < 1e-10);
}
