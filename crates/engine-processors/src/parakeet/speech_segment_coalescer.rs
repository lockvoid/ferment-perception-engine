//! Pad, merge, and clamp caller-supplied speech segments before decoding.

use ferment_engine_core::AudioEventSegment;

pub fn coalesce(
    segments: &[AudioEventSegment],
    pad_seconds: f64,
    duration_seconds: f64,
) -> Vec<AudioEventSegment> {
    let pad = pad_seconds.max(0.0);
    let upper = duration_seconds.max(0.0);
    let mut padded = segments
        .iter()
        .filter_map(|segment| {
            let start = (segment.start_time - pad).clamp(0.0, upper);
            let end = (segment.end_time() + pad).clamp(start, upper);
            (end > start).then_some(AudioEventSegment {
                event: segment.event,
                start_time: start,
                duration: end - start,
                confidence: segment.confidence,
            })
        })
        .collect::<Vec<_>>();
    padded.sort_by(|left, right| left.start_time.total_cmp(&right.start_time));

    let mut output: Vec<AudioEventSegment> = Vec::new();
    for next in padded {
        if let Some(current) = output.last_mut()
            && next.start_time <= current.end_time()
        {
            let end = current.end_time().max(next.end_time());
            current.duration = end - current.start_time;
            current.confidence = current.confidence.max(next.confidence);
            continue;
        }
        output.push(next);
    }
    output
}
