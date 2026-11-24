//! Current file-export cancellation/ownership assertions with original source
//! files and real platform codecs. Mux observers forward all real writes and
//! inject faults only after the real container has finished.
#![cfg(any(target_os = "macos", target_os = "linux", target_os = "android"))]
use ferment_engine_core::{
    EngineError, MultiVideoEncoding, ProgressSink, VideoEncoding, VideoEncodingVariant,
    WorkActivity, WorkEvent, WorkPhase,
};
use ferment_engine_media::{
    AudioMuxTrack, EncodedPacket, MuxBackend, MuxOutput, MuxSession, MuxSettings, MuxTrackId,
    Muxer, VideoError, VideoMuxTrack,
};
use ferment_engine_processors::{VideoEncoder, platform_encode_backends};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Directory {
    root: PathBuf,
    outputs: PathBuf,
}
impl Directory {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "ferment-current-encoder-cleanup-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let outputs = root.join("outputs");
        std::fs::create_dir_all(&outputs).unwrap();
        Self { root, outputs }
    }
    fn path(&self, name: &str) -> PathBuf {
        self.outputs.join(name)
    }
    fn empty(&self) {
        assert_eq!(
            std::fs::read_dir(&self.outputs).unwrap().count(),
            0,
            "export left a final or partial file"
        );
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}
fn fixture(name: &str) -> PathBuf {
    let root = std::env::var_os("FERMENT_ENGINE_FIXTURES_DIR")
        .map(PathBuf::from)
        .expect("original pinned current fixtures required");
    let path = root.join("Video").join(name);
    assert!(
        path.is_file(),
        "missing original fixture {}",
        path.display()
    );
    path
}
fn encoder() -> VideoEncoder {
    VideoEncoder::new().with_backends(platform_encode_backends())
}
fn variants(d: &Directory) -> Vec<VideoEncodingVariant> {
    vec![
        VideoEncodingVariant {
            id: "proxy".into(),
            output_path: d.path("proxy.mov"),
            preset: VideoEncoder::proxy_preset(),
            thumb_strip: None,
        },
        VideoEncodingVariant {
            id: "ai".into(),
            output_path: d.path("ai.mp4"),
            preset: VideoEncoder::ai_preset(),
            thumb_strip: None,
        },
    ]
}

/// Kill: retain the first committed branch when the second real output path cannot be created.
#[test]
#[ignore = "original reel-1/sec1.mp4 and real platform codecs"]
fn a_later_unwritable_branch_removes_every_output_owned_by_the_run() {
    let d = Directory::new();
    let source = fixture("reel-1/sec1.mp4");
    let blocker = d.root.join("not-a-directory");
    std::fs::write(&blocker, b"unrelated sentinel").unwrap();
    let mut requested = variants(&d);
    requested[1].output_path = blocker.join("ai.mp4");
    let result = futures_executor::block_on(encoder().encode_variants(
        &source,
        &requested,
        ProgressSink::noop(),
        None,
    ));
    assert!(result.is_err(), "unwritable second branch succeeded");
    d.empty();
    assert_eq!(std::fs::read(blocker).unwrap(), b"unrelated sentinel");
}

struct CancelOnFrame {
    progress: ProgressSink,
    slot: Arc<Mutex<Option<ProgressSink>>>,
    frames: Arc<AtomicUsize>,
}
impl CancelOnFrame {
    fn new() -> Self {
        let slot = Arc::new(Mutex::new(None::<ProgressSink>));
        let captured = Arc::clone(&slot);
        let frames = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&frames);
        let progress = ProgressSink::from_callback(move |event| {
            if event.stage == "Processing video" {
                let cancel = captured.lock().unwrap().take();
                if let Some(cancel) = cancel {
                    observed.fetch_add(1, Ordering::SeqCst);
                    cancel.cancel();
                }
            }
        });
        *slot.lock().unwrap() = Some(progress.clone());
        Self {
            progress,
            slot,
            frames,
        }
    }
}
impl Drop for CancelOnFrame {
    fn drop(&mut self) {
        self.slot.lock().unwrap().take();
    }
}

/// Kill: ignore cancellation delivered by the first actual completed-frame callback and leave either output behind.
#[test]
#[ignore = "original djs-walking-city.mp4 and real platform codecs; fresh process"]
fn cancellation_after_a_completed_frame_removes_both_variants_and_their_partials() {
    let d = Directory::new();
    let cancel = CancelOnFrame::new();
    let result = futures_executor::block_on(encoder().encode_variants(
        &fixture("djs-walking-city.mp4"),
        &variants(&d),
        cancel.progress.clone(),
        None,
    ));
    assert_eq!(
        cancel.frames.load(Ordering::SeqCst),
        1,
        "actual frame callback was not reached"
    );
    assert!(matches!(result, Err(EngineError::Cancelled)), "{result:?}");
    d.empty();
}

/// Kill: deliver the single proxy after its completed-frame callback cancels the actual sink.
#[test]
#[ignore = "original reel-1/sec1.mp4 and real platform codecs; fresh process"]
fn cancellation_after_a_completed_frame_never_delivers_the_single_proxy() {
    let d = Directory::new();
    let cancel = CancelOnFrame::new();
    let result = futures_executor::block_on(encoder().encode(
        &fixture("reel-1/sec1.mp4"),
        &d.path("proxy.mov"),
        &VideoEncoder::proxy_preset(),
        None,
        cancel.progress.clone(),
        None,
    ));
    assert_eq!(
        cancel.frames.load(Ordering::SeqCst),
        1,
        "actual frame callback was not reached"
    );
    assert!(matches!(result, Err(EngineError::Cancelled)), "{result:?}");
    d.empty();
}

fn assert_original(path: &Path, original: Option<&[u8]>) {
    match original {
        Some(bytes) => assert_eq!(
            std::fs::read(path).unwrap().as_slice(),
            bytes,
            "in-flight export changed original destination bytes"
        ),
        None => assert!(!path.exists(), "in-flight export exposed its destination"),
    }
}

#[derive(Clone, Copy, Debug)]
enum Fault {
    Observe,
    FailSecondFinishedMux,
    CancelFirstFinishedMux,
}

#[derive(Debug, Default)]
struct WorkAudit {
    phases: Vec<(u64, WorkPhase)>,
    checkpoints: usize,
    packet_writes: usize,
    missing_write_activity: usize,
    finalizers: usize,
    incorrectly_phased_finalizers: usize,
}
impl WorkAudit {
    fn record(&mut self, event: WorkEvent) {
        match event {
            WorkEvent::Started {
                id,
                phase: Some(phase),
            } => self.phases.push((id, phase)),
            WorkEvent::Started { phase: None, .. } => {}
            WorkEvent::Finished { id } => self.phases.retain(|(active, _)| *active != id),
            WorkEvent::Checkpoint => self.checkpoints += 1,
        }
    }
}

#[derive(Debug)]
struct WatchingMux {
    inner: MuxBackend,
    finals: Vec<PathBuf>,
    originals: Vec<Option<Vec<u8>>>,
    progress: ProgressSink,
    fault: Fault,
    opens: Arc<AtomicUsize>,
    finished: Arc<AtomicUsize>,
    work: Option<Arc<Mutex<WorkAudit>>>,
}
impl Muxer for WatchingMux {
    fn open(
        &self,
        path: &Path,
        settings: MuxSettings,
    ) -> ferment_engine_media::Result<Box<dyn MuxSession>> {
        let ordinal = self.opens.fetch_add(1, Ordering::SeqCst);
        let final_path = self
            .finals
            .get(ordinal)
            .expect("unexpected extra mux output")
            .clone();
        assert_ne!(
            path, final_path,
            "the real mux received the public destination"
        );
        assert_original(&final_path, self.originals[ordinal].as_deref());
        let inner = self.inner.open(path, settings)?;
        assert_original(&final_path, self.originals[ordinal].as_deref());
        Ok(Box::new(WatchingSession {
            inner,
            final_path,
            original: self.originals[ordinal].clone(),
            partial: path.to_owned(),
            progress: self.progress.clone(),
            fault: self.fault,
            ordinal,
            finished: self.finished.clone(),
            work: self.work.clone(),
            previous_write_start: None,
        }))
    }
}
#[derive(Debug)]
struct WatchingSession {
    inner: Box<dyn MuxSession>,
    final_path: PathBuf,
    original: Option<Vec<u8>>,
    partial: PathBuf,
    progress: ProgressSink,
    fault: Fault,
    ordinal: usize,
    finished: Arc<AtomicUsize>,
    work: Option<Arc<Mutex<WorkAudit>>>,
    previous_write_start: Option<usize>,
}
impl MuxSession for WatchingSession {
    fn add_video_track(
        &mut self,
        track: VideoMuxTrack,
    ) -> ferment_engine_media::Result<MuxTrackId> {
        self.inner.add_video_track(track)
    }
    fn add_audio_track(
        &mut self,
        track: AudioMuxTrack,
    ) -> ferment_engine_media::Result<MuxTrackId> {
        self.inner.add_audio_track(track)
    }
    fn write_packet(
        &mut self,
        track: MuxTrackId,
        packet: EncodedPacket,
    ) -> ferment_engine_media::Result<()> {
        if let Some(work) = &self.work {
            let mut audit = work.lock().unwrap();
            if self
                .previous_write_start
                .is_some_and(|before| audit.checkpoints <= before)
            {
                audit.missing_write_activity += 1;
            }
            self.previous_write_start = Some(audit.checkpoints);
        }
        assert_original(&self.final_path, self.original.as_deref());
        self.inner.write_packet(track, packet)?;
        assert_original(&self.final_path, self.original.as_deref());
        if let Some(work) = &self.work {
            work.lock().unwrap().packet_writes += 1;
        }
        Ok(())
    }
    fn finalize(self: Box<Self>) -> ferment_engine_media::Result<MuxOutput> {
        if let Some(work) = &self.work {
            let mut audit = work.lock().unwrap();
            audit.finalizers += 1;
            if audit.phases.last().map(|(_, phase)| *phase) != Some(WorkPhase::Finalizing) {
                audit.incorrectly_phased_finalizers += 1;
            }
            if self
                .previous_write_start
                .is_some_and(|before| audit.checkpoints <= before)
            {
                audit.missing_write_activity += 1;
            }
        }
        assert_original(&self.final_path, self.original.as_deref());
        let result = self.inner.finalize()?;
        assert!(self.partial.is_file());
        assert!(std::fs::metadata(&self.partial).unwrap().len() > 1024);
        assert_original(&self.final_path, self.original.as_deref());
        self.finished.fetch_add(1, Ordering::SeqCst);
        match (self.fault, self.ordinal) {
            (Fault::FailSecondFinishedMux, 1) => Err(VideoError::InvalidData(
                "fault after second real container finalized".into(),
            )),
            (Fault::CancelFirstFinishedMux, 0) => {
                self.progress.cancel();
                Ok(result)
            }
            _ => Ok(result),
        }
    }
}
fn watching(
    d: &Directory,
    progress: ProgressSink,
    fault: Fault,
    multi: bool,
) -> (VideoEncoder, Arc<AtomicUsize>) {
    watching_work(d, progress, fault, multi, None)
}

fn watching_work(
    d: &Directory,
    progress: ProgressSink,
    fault: Fault,
    multi: bool,
    work: Option<Arc<Mutex<WorkAudit>>>,
) -> (VideoEncoder, Arc<AtomicUsize>) {
    let mut backends = platform_encode_backends();
    let finished = Arc::new(AtomicUsize::new(0));
    let finals = if multi {
        vec![d.path("proxy.mov"), d.path("ai.mp4")]
    } else {
        vec![d.path("proxy.mov")]
    };
    backends.muxer = Arc::new(WatchingMux {
        inner: backends.muxer.clone(),
        originals: finals.iter().map(|p| std::fs::read(p).ok()).collect(),
        finals,
        progress,
        fault,
        opens: Arc::new(AtomicUsize::new(0)),
        finished: finished.clone(),
        work,
    });
    (VideoEncoder::new().with_backends(backends), finished)
}

/// Kill: omit successful packet checkpoints or enter finalization by display text instead of the actual mux call.
#[test]
#[ignore = "original reel-1/sec1.mp4 and real platform codecs"]
fn real_container_writes_report_activity_and_finalize_in_the_typed_phase() {
    let d = Directory::new();
    let audit = Arc::new(Mutex::new(WorkAudit::default()));
    let captured = Arc::clone(&audit);
    let work = WorkActivity::from_callback(move |event| captured.lock().unwrap().record(event));
    let progress = ProgressSink::noop().with_work_activity(work);
    let (encoder, finished) = watching_work(
        &d,
        progress.clone(),
        Fault::Observe,
        false,
        Some(Arc::clone(&audit)),
    );
    let output = d.path("proxy.mov");
    let result = futures_executor::block_on(encoder.encode(
        &fixture("reel-1/sec1.mp4"),
        &output,
        &VideoEncoder::proxy_preset(),
        None,
        progress,
        None,
    ))
    .unwrap();
    assert_eq!(finished.load(Ordering::SeqCst), 1);
    let audit = audit.lock().unwrap();
    assert!(
        audit.packet_writes > 1,
        "the real mux must write multiple actual packets"
    );
    assert_eq!(
        audit.missing_write_activity, 0,
        "completed mux writes were invisible to the host clock"
    );
    assert_eq!(audit.finalizers, 1);
    assert_eq!(audit.incorrectly_phased_finalizers, 0);
    assert!(
        audit.phases.is_empty(),
        "a completed call retained a logical phase"
    );
    assert_eq!(std::fs::metadata(&output).unwrap().len(), result.bytes);
    assert_eq!(std::fs::read_dir(&d.outputs).unwrap().count(), 1);
}

/// Kill: expose the requested URL during actual mux open, any packet write or finalization.
#[test]
#[ignore = "original reel-1/sec1.mp4 and real platform codecs"]
fn the_public_url_stays_absent_until_the_real_container_is_complete() {
    let d = Directory::new();
    let source = fixture("reel-1/sec1.mp4");
    let output = d.path("proxy.mov");
    let (encoder, finished) = watching(&d, ProgressSink::noop(), Fault::Observe, false);
    let result = futures_executor::block_on(encoder.encode(
        &source,
        &output,
        &VideoEncoder::proxy_preset(),
        None,
        ProgressSink::noop(),
        None,
    ))
    .unwrap();
    assert_eq!(finished.load(Ordering::SeqCst), 1);
    assert!(output.exists());
    assert!(result.bytes > 1024);
    assert_eq!(std::fs::read_dir(&d.outputs).unwrap().count(), 1);
    let info = futures_executor::block_on(platform_encode_backends().video.probe(&output)).unwrap();
    assert_eq!(info.video_streams().count(), 1);
}

/// Kill: roll back only partial files when a second actual mux fails after the first variant committed.
#[test]
#[ignore = "original reel-1/sec1.mp4 and real platform codecs"]
fn a_failure_after_the_second_real_mux_finishes_rolls_back_the_first_variant() {
    let d = Directory::new();
    let source = fixture("reel-1/sec1.mp4");
    let (encoder, finished) =
        watching(&d, ProgressSink::noop(), Fault::FailSecondFinishedMux, true);
    let result = futures_executor::block_on(encoder.encode_variants(
        &source,
        &variants(&d),
        ProgressSink::noop(),
        None,
    ));
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("second real container")
    );
    assert_eq!(finished.load(Ordering::SeqCst), 2);
    d.empty();
}

/// Kill: commit after a real mux returns even though cancellation arrived during its finalization.
#[test]
#[ignore = "original reel-1/sec1.mp4 and real platform codecs"]
fn cancellation_during_real_mux_finalization_never_promotes_the_finished_partial() {
    let d = Directory::new();
    let source = fixture("reel-1/sec1.mp4");
    let progress = ProgressSink::noop();
    let (encoder, finished) = watching(&d, progress.clone(), Fault::CancelFirstFinishedMux, false);
    let result = futures_executor::block_on(encoder.encode(
        &source,
        &d.path("proxy.mov"),
        &VideoEncoder::proxy_preset(),
        None,
        progress,
        None,
    ));
    assert!(matches!(result, Err(EngineError::Cancelled)), "{result:?}");
    assert_eq!(finished.load(Ordering::SeqCst), 1);
    d.empty();
}

/// Kill: delete or replace caller-owned destination bytes when a later branch cannot open its output.
#[test]
#[ignore = "original reel-1/sec1.mp4 and real platform codecs"]
fn a_later_branch_failure_preserves_the_preexisting_valid_movie_bytes() {
    let d = Directory::new();
    let source = fixture("reel-1/sec1.mp4");
    let original = std::fs::read(&source).unwrap();
    std::fs::write(d.path("proxy.mov"), &original).unwrap();
    let blocker = d.root.join("not-a-directory");
    std::fs::write(&blocker, b"owned blocker").unwrap();
    let mut requested = variants(&d);
    requested[1].output_path = blocker.join("ai.mp4");
    assert!(
        futures_executor::block_on(encoder().encode_variants(
            &source,
            &requested,
            ProgressSink::noop(),
            None
        ))
        .is_err()
    );
    assert_eq!(std::fs::read(d.path("proxy.mov")).unwrap(), original);
    assert_eq!(std::fs::read_dir(&d.outputs).unwrap().count(), 1);
}

/// Kill: overwrite an original movie when cancellation lands inside the real mux finalizer.
#[test]
#[ignore = "original reel-1/sec1.mp4 and real platform codecs"]
fn finalizer_cancellation_preserves_the_preexisting_valid_movie_bytes() {
    let d = Directory::new();
    let source = fixture("reel-1/sec1.mp4");
    let original = std::fs::read(&source).unwrap();
    std::fs::write(d.path("proxy.mov"), &original).unwrap();
    let progress = ProgressSink::noop();
    let (encoder, finished) = watching(&d, progress.clone(), Fault::CancelFirstFinishedMux, false);
    let result = futures_executor::block_on(encoder.encode(
        &source,
        &d.path("proxy.mov"),
        &VideoEncoder::proxy_preset(),
        None,
        progress,
        None,
    ));
    assert!(matches!(result, Err(EngineError::Cancelled)), "{result:?}");
    assert_eq!(finished.load(Ordering::SeqCst), 1);
    assert_eq!(std::fs::read(d.path("proxy.mov")).unwrap(), original);
    assert_eq!(std::fs::read_dir(&d.outputs).unwrap().count(), 1);
}

/// Kill: discard the first destination's backup when a later final destination cannot be installed.
#[test]
#[ignore = "original reel-1/sec1.mp4 and real platform codecs"]
fn a_later_commit_failure_restores_the_preexisting_first_movie() {
    let d = Directory::new();
    let source = fixture("reel-1/sec1.mp4");
    let original = std::fs::read(&source).unwrap();
    std::fs::write(d.path("proxy.mov"), &original).unwrap();
    std::fs::create_dir(d.path("ai.mp4")).unwrap();
    std::fs::write(d.path("ai.mp4").join("sentinel"), b"untouched directory").unwrap();
    let result = futures_executor::block_on(encoder().encode_variants(
        &source,
        &variants(&d),
        ProgressSink::noop(),
        None,
    ));
    assert!(result.is_err());
    assert_eq!(std::fs::read(d.path("proxy.mov")).unwrap(), original);
    assert_eq!(
        std::fs::read(d.path("ai.mp4").join("sentinel")).unwrap(),
        b"untouched directory"
    );
    assert_eq!(
        std::fs::read_dir(&d.outputs).unwrap().count(),
        2,
        "owned stage/backup leaked"
    );
}
