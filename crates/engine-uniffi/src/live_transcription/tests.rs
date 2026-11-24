use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Default)]
struct Observed {
    updates: Vec<Vec<f32>>,
    finishes: Vec<Vec<f32>>,
    cancels: usize,
}

struct Session {
    observed: Arc<Mutex<Observed>>,
    drops: Arc<AtomicUsize>,
    fail_update: bool,
    fail_finish: bool,
}

fn samples(source: &dyn AudioSampleSource) -> Vec<f32> {
    let mut result = vec![0.0; source.sample_count()];
    let count = result.len();
    assert_eq!(source.copy_samples(&mut result, 0, count).unwrap(), count);
    result
}

#[async_trait::async_trait(?Send)]
impl core::LiveTranscriptionSession for Session {
    async fn update(
        &mut self,
        source: &dyn AudioSampleSource,
    ) -> core::Result<core::LiveTranscript> {
        self.observed.lock().unwrap().updates.push(samples(source));
        if std::mem::take(&mut self.fail_update) {
            return Err(core::EngineError::inference("inference failed"));
        }
        Ok(core::LiveTranscript::default())
    }

    async fn finish(
        &mut self,
        source: &dyn AudioSampleSource,
    ) -> core::Result<core::TranscriptionResult> {
        self.observed.lock().unwrap().finishes.push(samples(source));
        if self.fail_finish {
            return Err(core::EngineError::inference("finish failed"));
        }
        Ok(core::TranscriptionResult {
            transcription: String::new(),
            language: None,
            captions: Vec::new(),
        })
    }

    async fn cancel(&mut self) {
        self.observed.lock().unwrap().cancels += 1;
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

fn session(
    fail_update: bool,
    fail_finish: bool,
) -> (
    LiveTranscriptionSession,
    Arc<Mutex<Observed>>,
    Arc<AtomicUsize>,
) {
    let observed = Arc::new(Mutex::new(Observed::default()));
    let drops = Arc::new(AtomicUsize::new(0));
    let session = LiveTranscriptionSession::new(Box::new(Session {
        observed: observed.clone(),
        drops: drops.clone(),
        fail_update,
        fail_finish,
    }));
    (session, observed, drops)
}

fn pcm(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Kill: replace the growing take with each delta, resend its prefix, or use
/// native/big endian at the wire boundary. Kotlin's existing test sends these bytes.
#[test]
fn append_only_wire_preserves_the_whole_take_and_finish_tail() {
    let (session, observed, drops) = session(false, false);
    session.update(pcm(&[0.5, -1.25])).unwrap();
    session.update(pcm(&[3.0])).unwrap();
    session.finish(pcm(&[0.75])).unwrap();
    assert_eq!(
        observed.lock().unwrap().updates,
        [vec![0.5, -1.25], vec![0.5, -1.25, 3.0]]
    );
    assert_eq!(
        observed.lock().unwrap().finishes,
        [vec![0.5, -1.25, 3.0, 0.75]]
    );
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(session.update(Vec::new()).is_err());
    assert!(session.finish(Vec::new()).is_err());
}

/// Kill: append a valid prefix before detecting a malformed/non-finite tail.
#[test]
fn malformed_pcm_never_partially_mutates_the_take() {
    let (session, observed, _) = session(false, false);
    session.update(pcm(&[0.5])).unwrap();
    let mut partial = pcm(&[1.0]);
    partial.push(3);
    assert!(session.update(partial).is_err());
    assert!(session.finish(pcm(&[1.0, f32::NAN])).is_err());
    session.finish(Vec::new()).unwrap();
    assert_eq!(observed.lock().unwrap().finishes, [vec![0.5]]);
}

/// Kill: discard already acknowledged PCM on a decoder failure, or keep the
/// native model alive after a terminal failure/cancel.
#[test]
fn retry_keeps_accepted_audio_and_terminal_paths_release_ownership() {
    let (failed, observed, drops) = session(true, true);
    assert!(failed.update(pcm(&[0.5])).is_err());
    failed.update(Vec::new()).unwrap();
    assert_eq!(observed.lock().unwrap().updates, [vec![0.5], vec![0.5]]);
    assert!(failed.finish(Vec::new()).is_err());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    failed.cancel();
    assert_eq!(observed.lock().unwrap().cancels, 0);

    let (cancelled, observed, drops) = session(false, false);
    cancelled.update(pcm(&[1.0])).unwrap();
    cancelled.cancel();
    cancelled.cancel();
    assert_eq!(observed.lock().unwrap().cancels, 1);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(cancelled.update(Vec::new()).is_err());
}
