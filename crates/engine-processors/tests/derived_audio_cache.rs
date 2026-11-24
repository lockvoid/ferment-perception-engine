//! The current iOS seven PCM cache cases, three source-factory cases and
//! five real processor cache cases. Explicit pinned fixtures/models required.
#![cfg(not(target_arch = "wasm32"))]
use ferment_engine_audio::derived_cache::{DerivedFileCache, DerivedSource, nonce};
use ferment_engine_audio::streaming_resampler::resample_current;
use ferment_engine_core::{AudioMinimapComposing, ProgressSink};
use ferment_engine_processors::{
    AudioMinimapComposer, AudioSampleSource, AudioSource, WaveformExtractor,
};
use futures_executor::block_on;
use std::fs::{self, File, FileTimes};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};

struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!("front-end-{}", nonce())))
    }
    fn cache(&self) -> Arc<DerivedFileCache> {
        Arc::new(DerivedFileCache::new(self.0.clone(), 1 << 30).unwrap())
    }
    fn files(&self, ext: &str) -> Vec<PathBuf> {
        fn walk(p: &Path, e: &str) -> Vec<PathBuf> {
            if !p.exists() {
                return vec![];
            }
            fs::read_dir(p)
                .unwrap()
                .flat_map(|x| {
                    let p = x.unwrap().path();
                    if p.is_dir() {
                        walk(&p, e)
                    } else if p.extension().is_some_and(|v| v == e) {
                        vec![p]
                    } else {
                        vec![]
                    }
                })
                .collect()
        }
        walk(&self.0, ext)
    }
}
impl Drop for Root {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn track() -> PathBuf {
    assert!(
        std::env::var_os("FERMENT_ENGINE_FIXTURES_DIR").is_some(),
        "explicit immutable fixtures required"
    );
    ferment_engine_test_fixtures::require_path("techno-1-30s.mp3")
}
fn load(
    path: &Path,
    rate: u32,
    cache: Option<&DerivedFileCache>,
) -> ferment_engine_audio::AudioFrames {
    block_on(AudioSource::default().resampled(path, rate, None, cache)).unwrap()
}
fn source_samples(source: &impl AudioSampleSource) -> Vec<f32> {
    let mut values = vec![0.; source.sample_count()];
    source
        .copy_samples(&mut values, 0, source.sample_count())
        .unwrap();
    values
}
fn inode(path: &Path) -> u64 {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        path.metadata().unwrap().ino()
    }
    #[cfg(not(unix))]
    {
        path.metadata().unwrap().len()
    }
}
fn mark_old(path: &Path) {
    File::options()
        .write(true)
        .open(path)
        .unwrap()
        .set_times(FileTimes::new().set_modified(UNIX_EPOCH + Duration::from_secs(1)))
        .unwrap();
}
fn sidecar(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&fs::read(path.with_extension("json")).unwrap()).unwrap()
}

#[test]
#[ignore = "requires pinned 30s MP3 and native decoder"]
fn cached_decode_is_bit_identical_to_direct_decode() {
    let r = Root::new();
    let c = r.cache();
    let plain = load(&track(), 22050, None);
    let cached = load(&track(), 22050, Some(&c));
    assert_eq!(cached.samples, plain.samples);
    assert_eq!(cached.sample_rate, 22050);
    assert!((cached.duration_secs() - 30.).abs() < 0.1);
    let pcm = r.files("pcm");
    assert_eq!(pcm.len(), 1);
    assert_eq!(
        pcm[0].metadata().unwrap().len(),
        cached.samples.len() as u64 * 4
    );
    assert_eq!(c.total_bytes(), cached.samples.len() as u64 * 4);
}
#[test]
#[ignore = "requires pinned 30s MP3 and native decoder"]
fn second_load_hits_the_same_inode_and_touches_it() {
    let r = Root::new();
    let c = r.cache();
    let first = load(&track(), 22050, Some(&c));
    let p = r.files("pcm")[0].clone();
    let id = inode(&p);
    mark_old(&p);
    let second = load(&track(), 22050, Some(&c));
    assert_eq!(first.samples, second.samples);
    assert_eq!(inode(&p), id);
    assert!(p.metadata().unwrap().modified().unwrap() > UNIX_EPOCH + Duration::from_secs(1));
    assert_eq!(r.files("pcm").len(), 1);
}
#[test]
#[ignore = "requires pinned 30s MP3 and native decoder"]
fn reopening_cache_and_moving_input_keeps_the_hit() {
    let r = Root::new();
    let first = load(&track(), 22050, Some(&r.cache()));
    let p = r.files("pcm")[0].clone();
    let id = inode(&p);
    let moved = r.0.join("moved.mp3");
    fs::copy(track(), &moved).unwrap();
    let second = load(&moved, 22050, Some(&r.cache()));
    assert_eq!(first.samples, second.samples);
    assert_eq!(inode(&p), id);
    assert_eq!(r.files("pcm").len(), 1);
}
#[test]
#[ignore = "requires pinned 30s MP3 and native decoder"]
fn rates_have_separate_artifacts_and_source_removal_invalidates_both() {
    let r = Root::new();
    let c = r.cache();
    let a = load(&track(), 22050, Some(&c));
    let b = load(&track(), 16000, Some(&c));
    assert_ne!(a.identity, b.identity);
    assert_eq!(r.files("pcm").len(), 2);
    c.remove_source(&DerivedSource::from_file(&track()).unwrap())
        .unwrap();
    assert!(r.files("pcm").is_empty());
    assert_eq!(c.total_bytes(), 0);
}
#[test]
#[ignore = "requires pinned 30s MP3 and native decoder"]
fn supported_rates_use_current_fir_or_polyphase_and_odd_rate_keeps_converter() {
    let native = block_on(AudioSource::default().decode(&track())).unwrap();
    assert_eq!(native.sample_rate, 44100);
    for rate in [22050, 16000] {
        assert_eq!(
            load(&track(), rate, None).samples,
            resample_current(&native.samples, 44100, rate).unwrap()
        );
    }
    let odd = load(&track(), 16001, None);
    let disk = block_on(AudioSource::default().decode_disk_backed(&track(), 16001)).unwrap();
    assert_eq!(odd.samples, source_samples(&disk));
    assert!((odd.duration_secs() - 30.).abs() < 0.1);
}
#[test]
#[ignore = "requires pinned 30s MP3 and native decoder"]
fn absent_cache_does_not_create_a_cache_directory() {
    let r = Root::new();
    let a = load(&track(), 22050, None);
    let b = load(&track(), 22050, None);
    assert_eq!(a.samples, b.samples);
    assert!(!r.0.exists());
}
#[test]
#[ignore = "requires pinned 30s MP3 and native decoder"]
fn resampled_identity_is_its_request_hash_with_or_without_a_cache() {
    let r = Root::new();
    let c = r.cache();
    let a = load(&track(), 22050, Some(&c));
    let b = load(&track(), 22050, None);
    assert_eq!(a.identity, b.identity);
    assert_eq!(
        a.identity.unwrap().id,
        format!(
            "pcm:{}",
            r.files("pcm")[0].file_stem().unwrap().to_str().unwrap()
        )
    );
    assert_ne!(b.identity, load(&track(), 16000, None).identity);
    assert!(
        block_on(AudioSource::default().decode(&track()))
            .unwrap()
            .identity
            .is_none()
    );
}
#[test]
#[ignore = "requires pinned 30s MP3 and native decoder"]
fn disk_source_streams_the_same_bytes_as_flat_source() {
    for rate in [22050, 16000] {
        let disk = block_on(AudioSource::default().decode_disk_backed(&track(), rate)).unwrap();
        assert_eq!(source_samples(&disk), load(&track(), rate, None).samples);
        let p = disk.path().to_owned();
        disk.cleanup().unwrap();
        assert!(!p.exists());
    }
}
#[test]
#[ignore = "requires pinned 30s MP3 and native decoder"]
fn unsupported_ratio_still_streams_full_duration() {
    let disk = block_on(AudioSource::default().decode_disk_backed(&track(), 16001)).unwrap();
    assert!((disk.sample_count() as f64 / 16001. - 30.).abs() < 0.1);
}
#[test]
#[ignore = "requires pinned 30s MP3 and native decoder"]
fn cached_source_cleanup_leaves_cache_owned_file_and_a_second_reader_hits() {
    let r = Root::new();
    let c = r.cache();
    let a = block_on(AudioSource::default().cached_source(&track(), 16000, None, &c)).unwrap();
    let p = a.path().to_owned();
    let id = inode(&p);
    assert_eq!(source_samples(&a), load(&track(), 16000, None).samples);
    a.cleanup().unwrap();
    assert!(p.exists());
    let b = block_on(AudioSource::default().cached_source(&track(), 16000, None, &c)).unwrap();
    assert_eq!(source_samples(&a), source_samples(&b));
    assert_eq!(inode(&p), id);
    assert_eq!(c.total_bytes(), a.sample_count() as u64 * 4);
    c.remove_all().unwrap();
    assert!(!p.exists());
    assert_eq!(
        source_samples(&a),
        source_samples(&b),
        "already-open readers survive eviction"
    );
}
#[test]
#[ignore = "requires pinned 30s MP3 and native decoder"]
fn minimap_populates_pcm_and_mel_once_and_hits_both() {
    let r = Root::new();
    let c = r.cache();
    let composer = AudioMinimapComposer::with_defaults().with_cache(Some(c));
    let cold = block_on(AudioMinimapComposing::compose(
        &composer,
        &track(),
        &[],
        ProgressSink::noop(),
    ))
    .unwrap();
    let pcm = r.files("pcm")[0].clone();
    let mel = r.files("mel")[0].clone();
    let ids = (inode(&pcm), inode(&mel));
    let warm = block_on(AudioMinimapComposing::compose(
        &composer,
        &track(),
        &[],
        ProgressSink::noop(),
    ))
    .unwrap();
    assert_eq!((r.files("pcm").len(), r.files("mel").len()), (1, 1));
    assert_eq!(ids, (inode(&pcm), inode(&mel)));
    assert_eq!(warm.image_data, cold.image_data);
    assert_eq!(warm.width, cold.width);
    assert!(cold.image_data.len() > 10000);
    assert_eq!(
        mel.metadata().unwrap().len() / 4 / 128,
        pcm.metadata().unwrap().len() / 4 / 512 + 1
    );
}
#[test]
#[ignore = "requires pinned 30s MP3 and native decoder"]
fn waveform_populates_22050_pcm_and_hits_on_second_run() {
    let r = Root::new();
    let c = r.cache();
    let cold = block_on(WaveformExtractor.extract_with_cache(&track(), 0.01, Some(&c))).unwrap();
    let p = r.files("pcm")[0].clone();
    let id = inode(&p);
    assert!((p.metadata().unwrap().len() as f64 / 4. / 22050. - 30.).abs() < 0.1);
    let warm = block_on(WaveformExtractor.extract_with_cache(&track(), 0.01, Some(&c))).unwrap();
    assert_eq!(cold.rms, warm.rms);
    assert_eq!(cold.peaks, warm.peaks);
    assert_eq!(cold.duration, warm.duration);
    assert_eq!(inode(&p), id);
    assert_eq!(r.files("pcm").len(), 1);
}

#[cfg(feature = "ort")]
mod models {
    use super::*;
    use ferment_engine_core::{AudioSemanticEncoding, BeatAnalyzing, MusicHighlighting};
    use ferment_engine_infer::{
        Backend, ModelId, ModelPaths, ModelVersion, OrtBackend, OrtExecutionProvider,
    };
    use ferment_engine_processors::{AudioSemanticEncoder, BeatThis, MusicHighlighter};
    fn backend(id: ModelId, file: &str) -> Backend {
        assert!(
            std::env::var_os("FERMENT_ENGINE_MODELS_DIR").is_some(),
            "explicit pinned models required"
        );
        let path = ferment_engine_test_fixtures::export::artifact(file).expect("pinned model");
        Arc::new(OrtBackend::with_execution_provider(
            Arc::new(ModelPaths::new().with_path(id, ModelVersion::Latest, path)),
            OrtExecutionProvider::Cpu,
        ))
    }
    #[test]
    #[ignore = "requires pinned MP3, beat_this_1 and ORT"]
    fn beat_this_fills_both_cache_doors_once_and_hits_on_second_run() {
        let r = Root::new();
        let c = r.cache();
        let processor = BeatThis::new(backend(ModelId::BeatThis1, "beat_this_1.onnx"))
            .with_ensemble_size(1)
            .with_cache(Some(c));
        let cold = block_on(BeatAnalyzing::detect(
            &processor,
            &track(),
            ProgressSink::noop(),
        ))
        .unwrap();
        let pcm = r.files("pcm")[0].clone();
        let mel = r.files("mel")[0].clone();
        let ids = (inode(&pcm), inode(&mel));
        assert_ne!(pcm.parent(), mel.parent());
        assert_eq!(
            sidecar(&mel)["key"]["source"]["id"],
            format!("pcm:{}", pcm.file_stem().unwrap().to_str().unwrap())
        );
        mark_old(&pcm);
        mark_old(&mel);
        let warm = block_on(BeatAnalyzing::detect(
            &processor,
            &track(),
            ProgressSink::noop(),
        ))
        .unwrap();
        assert!(!cold.beats.is_empty());
        assert_eq!(cold.beats, warm.beats);
        assert_eq!(cold.downbeats, warm.downbeats);
        assert_eq!((r.files("pcm").len(), r.files("mel").len()), (1, 1));
        assert_eq!(ids, (inode(&pcm), inode(&mel)));
        for p in [&pcm, &mel] {
            assert!(
                p.metadata().unwrap().modified().unwrap() > UNIX_EPOCH + Duration::from_secs(1)
            );
        }
        assert_eq!(
            mel.metadata().unwrap().len() / 4 / 128,
            pcm.metadata().unwrap().len() / 4 / 441 + 1
        );
    }
    #[test]
    #[ignore = "requires pinned MP3, music_highlighter and ORT"]
    fn highlighter_shares_minimap_pcm_but_uses_its_own_mel() {
        let r = Root::new();
        let c = r.cache();
        block_on(AudioMinimapComposing::compose(
            &AudioMinimapComposer::with_defaults().with_cache(Some(c.clone())),
            &track(),
            &[],
            ProgressSink::noop(),
        ))
        .unwrap();
        let p = r.files("pcm")[0].clone();
        let id = inode(&p);
        let processor =
            MusicHighlighter::new(backend(ModelId::MusicHighlighter, "music_highlighter.onnx"))
                .with_cache(Some(c));
        let cold = block_on(MusicHighlighting::detect(
            &processor,
            &track(),
            &[10],
            ProgressSink::noop(),
        ))
        .unwrap();
        assert_eq!((r.files("pcm").len(), r.files("mel").len()), (1, 2));
        assert_eq!(inode(&p), id);
        let warm = block_on(MusicHighlighting::detect(
            &processor,
            &track(),
            &[10],
            ProgressSink::noop(),
        ))
        .unwrap();
        assert_eq!(cold.saliency, warm.saliency);
        assert_eq!(
            cold.highlights
                .iter()
                .map(|x| x.start_time)
                .collect::<Vec<_>>(),
            warm.highlights
                .iter()
                .map(|x| x.start_time)
                .collect::<Vec<_>>()
        );
        assert_eq!(r.files("mel").len(), 2);
    }
    #[test]
    #[ignore = "requires pinned MP3, dasheng and ORT"]
    fn dasheng_encodes_from_cached_sixteen_kilohertz_pcm() {
        let r = Root::new();
        let c = r.cache();
        let processor =
            AudioSemanticEncoder::new(backend(ModelId::DashengBase, "dasheng_base_mel.onnx"))
                .with_cache(Some(c));
        let cold = block_on(AudioSemanticEncoding::encode(
            &processor,
            &track(),
            false,
            ProgressSink::noop(),
        ))
        .unwrap();
        let p = r.files("pcm")[0].clone();
        let id = inode(&p);
        assert!((p.metadata().unwrap().len() as f64 / 4. / 16000. - 30.).abs() < 0.1);
        let warm = block_on(AudioSemanticEncoding::encode(
            &processor,
            &track(),
            false,
            ProgressSink::noop(),
        ))
        .unwrap();
        assert_eq!(cold.pooled, warm.pooled);
        assert_eq!(inode(&p), id);
        assert_eq!(r.files("pcm").len(), 1);
        assert_eq!(r.files("mel").len(), 0);
    }
}
