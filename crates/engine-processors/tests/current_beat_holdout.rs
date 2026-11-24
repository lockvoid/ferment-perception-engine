//! The named Ballroom holdout supplied to the engine oracle.
use ferment_engine_processors::beatthis::{meter::MeterResult, quantize::quantize_beats};

#[test]
#[ignore = "requires the original Ballroom annotation in FERMENT_ENGINE_FIXTURES_DIR"]
fn current_worst_ballroom_holdout_keeps_its_beats() {
    let path = ferment_engine_test_fixtures::require_path("Ballroom/Albums-Latin_Jam5-01.beats");
    let text = std::fs::read_to_string(path).expect("read the pinned Ballroom annotation");
    let mut beats = Vec::new();
    let mut downbeats = Vec::new();
    for line in text.lines() {
        let columns: Vec<_> = line.split_whitespace().collect();
        assert_eq!(columns.len(), 2);
        let time = columns[0].parse::<f64>().unwrap();
        let in_bar = columns[1].parse::<usize>().unwrap();
        beats.push(time);
        if in_bar == 1 {
            downbeats.push(time);
        }
    }
    assert_eq!(beats.len(), 50, "current iOS named original holdout");
    let result = quantize_beats(
        &beats,
        &downbeats,
        Some(MeterResult {
            meter: 4,
            phase: 0,
            raw_meter: 4,
        }),
        None,
    );
    let q = result.beats;
    assert!(q.len() as f64 >= beats.len() as f64 * 0.95);
    let mut j = 0;
    let mut worst = 0.0_f64;
    for beat in beats {
        while j + 1 < q.len() && (q[j + 1] - beat).abs() <= (q[j] - beat).abs() {
            j += 1;
        }
        worst = worst.max((q[j] - beat).abs());
    }
    assert!(worst < 0.30, "orphaned beat at a segment join: {worst}");
}
