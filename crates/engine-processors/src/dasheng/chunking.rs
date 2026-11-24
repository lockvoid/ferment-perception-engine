//! Non-overlapping 10-second window chunking with zero-padding for the
//! tail. Matches iOS `AudioSemanticEncoder.chunkIntoWindows` exactly.

use super::config::WINDOW_SAMPLES;

/// Split mono audio into non-overlapping `WINDOW_SAMPLES`-sized chunks.
/// The final chunk is zero-padded if shorter. Always returns at least one
/// chunk (a fully-zero one for empty input).
pub fn chunk_into_windows(audio: &[f32]) -> Vec<Vec<f32>> {
    if audio.len() <= WINDOW_SAMPLES {
        let mut chunk = Vec::with_capacity(WINDOW_SAMPLES);
        chunk.extend_from_slice(audio);
        chunk.resize(WINDOW_SAMPLES, 0.0);
        return vec![chunk];
    }

    let n = audio.len().div_ceil(WINDOW_SAMPLES);
    let mut chunks = Vec::with_capacity(n);
    let mut start = 0usize;
    while start < audio.len() {
        let end = (start + WINDOW_SAMPLES).min(audio.len());
        let mut chunk = Vec::with_capacity(WINDOW_SAMPLES);
        chunk.extend_from_slice(&audio[start..end]);
        if chunk.len() < WINDOW_SAMPLES {
            chunk.resize(WINDOW_SAMPLES, 0.0);
        }
        chunks.push(chunk);
        start += WINDOW_SAMPLES;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_yields_one_zero_window() {
        let chunks = chunk_into_windows(&[]);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), WINDOW_SAMPLES);
        assert!(chunks[0].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_chunk_shorter_than_window_pads_to_one_window() {
        let audio = vec![0.5f32; 1000];
        let chunks = chunk_into_windows(&audio);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), WINDOW_SAMPLES);
        assert_eq!(chunks[0][0], 0.5);
        assert_eq!(chunks[0][1000], 0.0);
        assert_eq!(chunks[0][WINDOW_SAMPLES - 1], 0.0);
    }

    #[test]
    fn test_chunk_exactly_window_yields_one_window() {
        let audio = vec![0.25f32; WINDOW_SAMPLES];
        let chunks = chunk_into_windows(&audio);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), WINDOW_SAMPLES);
        assert_eq!(chunks[0][0], 0.25);
        assert_eq!(chunks[0][WINDOW_SAMPLES - 1], 0.25);
    }

    #[test]
    fn test_chunk_multiple_whole_windows() {
        let chunks = chunk_into_windows(&vec![1.0; WINDOW_SAMPLES * 3]);
        assert_eq!(chunks.len(), 3);
        assert!(chunks.iter().all(|chunk| chunk.len() == WINDOW_SAMPLES));
        assert!(chunks.iter().flatten().all(|&sample| sample == 1.0));
    }

    #[test]
    fn test_chunk_partial_final_window_gets_zero_padded() {
        let chunks = chunk_into_windows(&vec![0.7; WINDOW_SAMPLES * 2 + 1_234]);
        let last = chunks.last().unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(last.len(), WINDOW_SAMPLES);
        assert_eq!(last[1_233], 0.7);
        assert_eq!(last[1_234], 0.0);
        assert_eq!(last[WINDOW_SAMPLES - 1], 0.0);
    }

    #[test]
    fn test_chunk_preserves_sample_order() {
        let count = WINDOW_SAMPLES * 2 + 100;
        let audio = (0..count).map(|index| index as f32).collect::<Vec<_>>();
        let chunks = chunk_into_windows(&audio);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0][0], 0.0);
        assert_eq!(chunks[0][WINDOW_SAMPLES - 1], (WINDOW_SAMPLES - 1) as f32);
        assert_eq!(chunks[1][0], WINDOW_SAMPLES as f32);
        assert_eq!(chunks[2][0], (WINDOW_SAMPLES * 2) as f32);
    }

    #[test]
    fn three_full_windows_then_padded_tail() {
        // 3.5 windows worth → 4 chunks
        let n = WINDOW_SAMPLES * 3 + WINDOW_SAMPLES / 2;
        let audio = vec![1.0f32; n];
        let chunks = chunk_into_windows(&audio);
        assert_eq!(chunks.len(), 4);
        for c in chunks.iter().take(3) {
            assert!(c.iter().all(|&v| v == 1.0));
        }
        // Last chunk: half ones, half zeros
        let last = &chunks[3];
        for &v in &last[..WINDOW_SAMPLES / 2] {
            assert_eq!(v, 1.0);
        }
        for &v in &last[WINDOW_SAMPLES / 2..] {
            assert_eq!(v, 0.0);
        }
    }
}
