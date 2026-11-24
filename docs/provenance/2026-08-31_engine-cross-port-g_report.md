# Track G report — Video encoder: deterministic rate control + variants

**Executed 2026-08-31 against `main @ 5b97e84c`** (shared working tree, tracks
B–F live alongside). Dispatch doc: `2026-08-31_engine-cross-port.md` § "Track G".
iOS reference: `ferment-cuts-ios @ 3d1c2c180`.

Everything the acceptance list asks for is landed and green. Three things the
dispatch doc did not anticipate turned up while proving it runs; they are in
**Deviations** and **Measured platform behaviour**.

---

## Zero-trust pass: every constant re-read from the Swift before coding

| Doc claim | Source | Verdict |
|---|---|---|
| `VideoRateControl { average(Int) \| bitsPerPixel(Double) }` | `Protocols/VideoEncoding.swift:18-41` | **Confirmed.** |
| `resolvedBitRate = bpp × w × h × fps` | same, `:38` | **Confirmed** — `Int(bpp * Double(width*height) * fps)`, truncating. |
| fps fallback 30 on 0/non-finite | same, `:37` | **Confirmed** — `fps.isFinite && fps > 0 ? fps : 30`. |
| proxy: bpp 0.75, cap 1080, `capsShortEdge`, bFrames 0, kfi 1, HDR-eligible, `.mov` | `:103-112` | **Confirmed, all six.** |
| ai: average 2 500 000, maxWidth 640, kfi 60, SDR, `.mp4` | `:140-148` | **Confirmed.** `capsShortEdge` defaults to `false`, so the AI feed caps the LONG edge. |
| ai reordering OFF explicitly | `VideoEncoder.swift:455-464`, commit `11b25b977` | **Confirmed** — upstream states `AVVideoAllowFrameReorderingKey = false` whenever `bFrames == 0`, with the comment "a preset with a GOP longer than one frame could silently regain B-frames". iOS reaches it via `bFrames: 0`, not via a separate flag. |
| `scaledSize` caps `min(w,h)` when `capsShortEdge` | `VideoEncoder.swift:575-583` | **Confirmed.** One divergence, deliberate — see Deviations. |
| No quality-mode probes anywhere | `ca4f3e6c2` deletes `VideoEncoderQualityProbeTests.swift` (85 LOC) and `AVVideoQualityKey` | **Confirmed.** Nothing in redshift ever asked for a quality; nothing does now. |
| HDR split: shared decode for HDR-preserving, SDR from the finished proxy | `VideoEncoderMultiVariant.swift` `encodeSplittingHDR`, commit `203deb5d9` | **Confirmed**, including "with no HDR variant in the request each SDR variant pays its own source decode". |
| Thumbstrip container format | `FermentCuts/Components/ThumbStrip/Data/ThumbContainer.swift:19-29` | **Confirmed** — `PCKT` v2, 24-byte header, 16-byte index slots, `validCount` as the publish point. |
| `withThumbStrip` composes from the shared decode | `EncodeVideoProcessor.swift:92-134` | **Confirmed** — 0.5 s interval, 192×192 cells, `maxCount = ceil(duration/interval)+1`, 2400 fallback, JPEG q80. |

**One doc phrasing the source contradicts.** The dispatch doc reads as if `ai`
needs a *separate* "reordering off" switch beside `bFrames: 0`. Upstream has no
such switch: `bFrames == 0` **is** the switch, and the explicit
`AllowFrameReordering = false` is what that branch emits. The port keeps the
same shape (`b_frames: 0` → `allow_frame_reordering: false`) and the assertion
the doc asks for is on the derived setting, which is where the bug would live.

---

## What landed

### Protocol layer — `crates/engine-core/src/capabilities.rs`

```rust
pub enum VideoRateControl { Average(u64), BitsPerPixel(f64) }
impl VideoRateControl {
    pub const FALLBACK_FRAME_RATE: f64 = 30.0;
    pub fn resolved_bit_rate(self, width: u32, height: u32, frames_per_second: f64) -> u64;
}

pub struct VideoEncodingPreset {
    pub label: String,
    pub file_type: VideoFileType,
    pub max_width: u32,
    pub caps_short_edge: bool,        // NEW
    pub rate_control: VideoRateControl,  // replaces `bit_rate: u32`
    pub b_frames: u8,
    pub key_frame_interval: u32,
    pub hdr_eligible: bool,
}

pub struct VideoEncodingVariant { pub id: String, pub output_path: PathBuf,
                                  pub preset: VideoEncodingPreset,
                                  pub thumb_strip: Option<ThumbStripOptions> }
pub struct MultiVideoEncodingResult { pub variants: BTreeMap<String, VideoEncodingResult>,
                                      pub wall_seconds: f64, pub decoded_frames: usize }
pub trait VariantThumbnailSink { fn emit(&self, variant_id: &str, thumbnail: ThumbnailFrame); }

#[async_trait(?Send)]
pub trait MultiVideoEncoding: EngineService {
    async fn encode_variants(&self, source_path: &Path, variants: &[VideoEncodingVariant],
                             progress: ProgressSink,
                             on_thumbnail: Option<&dyn VariantThumbnailSink>)
        -> Result<MultiVideoEncodingResult>;
}
```

`VideoEncoder` implements both `VideoEncoding` and `MultiVideoEncoding`.

### Encoder — `crates/engine-processors/src/video_encoder.rs`

```rust
pub struct SourceVideo { pub width: u32, pub height: u32,
                         pub frames_per_second: f64, pub color: ColorInfo }
impl SourceVideo { pub fn from_stream(stream: &StreamInfo) -> Result<Self>; }

pub struct VideoCodecSettings {
    pub codec: Codec, pub width: u32, pub height: u32,
    pub bit_rate: u64,                       // was u32
    pub profile: VideoProfile,
    pub max_key_frame_interval: Option<u32>,
    pub allow_frame_reordering: bool,
    pub color: ColorInfo,                    // always stated, never inferred
    pub insert_hdr_metadata: bool,
}

VideoEncoder::proxy_preset() / ai_preset()
VideoEncoder::scaled_size(w, h, max_edge, caps_short_edge) -> (u32, u32)
VideoEncoder::plan(source: SourceVideo, preset: &VideoEncodingPreset) -> VideoCodecSettings
VideoEncoder::{hdr,sdr}_video_settings(w, h, bit_rate, b_frames, key_frame_interval[, transfer])
VideoEncoder::validate_thumb_strip(..) / validate_variants(..)
```

### Pipeline — `crates/engine-processors/src/encode_pipeline.rs`

`shared_decode_encode` is the one decode loop; `transcode_video` (single
output) and `transcode_variants` (multi + HDR split) both go through it.
`VariantLane` owns one output's plan, encoder session, thumbnail cadence and
delivery. Every lane scales from the **same full-resolution decoded frame**, so
within a shared pass a smaller variant is not a second-generation copy of a
larger one (a deliberate improvement over iOS, which decodes at the lead's size
and scales the tail from it — noted in Deviations).

### Thumbstrip container — `crates/engine-processors/src/thumb_container.rs` (new)

`PCKT` v2, byte-for-byte the iOS layout:

```
[0..4)   "PCKT"
[4..8)   version u32 LE = 2
[8..12)  max_count u32 LE
[12..16) valid_count u32 LE      <- publish point, written last
[16..20) data_offset u32 LE = 24 + max_count*16
[20..24) reserved u32 LE = 0
[24..data_offset)  index slots (offset u64 LE, size u64 LE) * max_count
[data_offset..EOF) cell bytes, appended
```

API: `create_empty`, `open`, `open_or_create`, `valid_count`, `index_entry`,
`cell`, `append`, `reset`, `reserved_len`, `max_count`, plus
`encode_thumbnail_jpeg(&ThumbnailFrame) -> Vec<u8>` at q80. Write order is
bytes → slot → count; all I/O is positional (`read_at`/`write_at` on unix,
`seek_read`/`seek_write` on Windows) so a reader with its own handle never
shares a cursor with the writer.

---

## Per-platform knob table (measured, not assumed)

Everything below was exercised on this Mac through `encode_pipeline_e2e`.
The Windows column is documentation + a cross-target `cargo check`/`clippy`
against `x86_64-pc-windows-msvc` (both clean); **it has not been run on the
Windows box.**

| Preset knob | macOS / VideoToolbox | Windows / Media Foundation | Fallback if refused |
|---|---|---|---|
| average bit rate | `kVTCompressionPropertyKey_AverageBitRate`. Accepted for H.264 and HEVC. | `MF_MT_AVG_BITRATE` on the output type. The inbox H.264 MFT's default rate-control mode is unconstrained VBR targeting exactly that attribute (MS docs), so no `AVEncCommonRateControlMode` call is needed. | none — a refusal is an error, the encode fails. |
| max keyframe interval | `kVTCompressionPropertyKey_MaxKeyFrameInterval`. Accepted. | `MF_MT_MAX_KEYFRAME_SPACING` + `CODECAPI_AVEncMPVGOPSize` (pre-existing). | none. |
| **frame reordering off** | `kVTCompressionPropertyKey_AllowFrameReordering = false`. **Accepted; and load-bearing** — see the mutation table: dropping the call makes VideoToolbox emit composition-offset packets on the 12-frame fixture, so its documented `true` default is real, not theoretical. | `CODECAPI_AVEncMPVDefaultBPictureCount = 0`, set before `SetOutputType` as the MFT requires. | none. |
| frame reordering **on** | `AllowFrameReordering = true`. Accepted. | **Cannot be honoured.** `AVEncMPVDefaultBPictureCount` defaults to 0 and MS lists it as an *optional*, non-HCK-tested property, so "allow reordering" is a no-op: MF simply emits no B-frames. **No product impact** — both stock presets use `b_frames: 0`. Recorded rather than hidden. |
| profile | `kVTCompressionPropertyKey_ProfileLevel` → `H264_High_AutoLevel` / `HEVC_Main10_AutoLevel`. | `MF_MT_MPEG2_PROFILE` (66/77/100). **HEVC Main10 is unavailable** — the MF backend already refuses `Codec::Hevc` with `"HEVC encoding is unavailable on the inventoried Windows Server host"`. So the **HDR proxy has no Windows encoder**; the request errors instead of silently emitting SDR. |
| colour primaries / transfer / matrix | `kVTCompressionPropertyKey_{ColorPrimaries,TransferFunction,YCbCrMatrix}` from CoreVideo constants. **All accepted** for 709 and 2020/HLG/PQ; a `kVTPropertyReadOnlyErr` from an encoder that enforces its own colorimetry is surfaced, not swallowed. | **Deliberately not set on the media type.** MS's supported-attribute list for the inbox H.264 encoder has no colour attributes, and setting an unrecognised one risks `MF_E_INVALIDMEDIATYPE` on a box I cannot test. **This is not a dropped setting**: the tags ride the container — `VideoMuxTrack.color` is fed from the plan and the pure-Rust muxer writes the `colr`/`nclx` box on both platforms, which is what players read. macOS additionally gets them into the elementary-stream VUI. |
| HDR metadata insertion | `HDRMetadataInsertionMode = Auto` + `PreserveDynamicHDRMetadata = false`. **Accepted on HEVC. REFUSED on H.264 with `kVTParameterErr` (-12902)** — measured, see below. | **Refused up front** with `"Media Foundation H.264 encoding cannot insert HDR metadata"`. An H.264 stream with no HDR metadata but an HDR container tag is worse than an honest failure. |
| unsupported colour enum | `UnsupportedCodec("VideoToolbox color primaries …")` rather than an approximation. | n/a. | none. |

**Measured refusal, verbatim.** Asking an H.264 session for HDR metadata:

```
Platform { operation: "VTSessionSetProperty(HDRMetadataInsertionMode)", status: -12902 }
```

Found by a test that asked for it on both lanes. Production never does — only
`hdr_video_settings` (HEVC) sets the flag — but the refusal is now a documented
fact instead of an assumption, and the error surfaces rather than vanishing.

---

## Deviations from the dispatch doc, and why

1. **`crates/engine-core` and `crates/engine-media/src/frame.rs` were touched**,
   beyond the stated "video_encoder.rs, encode_pipeline.rs, engine-media
   platform encoder settings".
   - `engine-core`: `VideoRateControl` and `caps_short_edge` are *protocol*
     (`Protocols/VideoEncoding.swift`), and `VideoEncodingPreset` already lives
     there. The round's SCOPE puts `crates/engine-core` in bounds.
   - `engine-media/src/frame.rs`: **required, see item 2.**
   `crates/project`, `apps/`, `packages/` and all graph/cook wiring: untouched.
   `crates/project` still compiles (`cargo check -p ferment-project` clean) —
   it only calls `proxy_preset()`/`ai_preset()`, never constructs a preset.

2. **The HDR split could not run at all until `to_rgb24` learned 10-bit.**
   The first e2e run of the split failed with
   `unsupported codec: CPU conversion for CV pixel format 0x78343230` — `x420`,
   `kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange`. The HDR proxy is genuine
   HEVC Main10, so the SDR derivative's decode of it hits a format
   `MacPixelBuffer::to_rgb24` had no case for. Any true 10-bit HDR import was
   equally undecodable. Added a ~40-line `narrow_ten_bit_planes` that converts
   the planes to 8-bit codes and reuses the existing `nv12_to_rgb` colour math —
   no duplicated conversion logic. **This is a decoder change, not an encoder
   setting; flagging it explicitly for the lead.**

3. **Every lane scales from the full-resolution decoded frame.** iOS asks the
   `AVAssetReader` for the lead variant's size and `VTPixelTransfer`s the tail
   down from it, because the reader does the scaling on the GPU. redshift
   already decodes at native size and CPU-scales per output, so a "lead" would
   only add a generation of loss. The `VideoEncoderMultiVariant` handoff queue,
   `idleWakeups` and `MultiEncodeDiagnostics` are all artefacts of
   AVFoundation's `requestMediaDataWhenReady` pump and have no counterpart in a
   pull loop — **not ported, deliberately**. `MultiVideoEncodingResult` therefore
   carries `decoded_frames` but no `idle_wakeups`.

4. **`scaled_size` clamps to a minimum of 2 px per axis.** Upstream returns the
   raw `Int(w*scale) & ~1`, which is `0` for a sub-2-pixel edge and would hand
   the platform encoder a zero dimension. Pre-existing redshift behaviour, kept,
   and now covered by a test that names the reason.

5. **Thumb cells are 4:4:4, not 4:2:0.** iOS uses libjpeg-turbo `TJSAMP_420`;
   `image`'s baseline encoder is fixed at `h=1,v=1` for every component
   (`image-0.25.10/src/codecs/jpeg/encoder.rs:398-427`), so cells come out
   roughly 40% larger at the same quality. Chose the already-vendored `image`
   over adding `jpeg-encoder` because **`cargo deny check licenses` is currently
   red on the loro tree** (ReplicaMan's, not mine) and this round's rule is
   "stop and report, do not vendor around it" — adding a dependency into a red
   licence gate would bury the signal. The container stores opaque bytes, so
   swapping the codec later is a one-function change.

6. **`VideoEncodingPlan` was deleted during the review pass.** It wrapped
   `VideoCodecSettings` plus a `thumbnails_enabled: bool` that only tests read —
   the pipeline tracks its own per-lane strip. `plan()` now returns
   `VideoCodecSettings` directly and is **infallible**; thumb-strip validation
   stays where it belongs, in `validate_thumb_strip`/`validate_variants` at the
   API boundary. Removed the fake `has_thumbnail_handler: true` the pipeline had
   to pass.

7. **Seven `#[ignore]` "blocked" stubs deleted** from `tests/video_encoder.rs`
   (`…_playable_container_assertions` ×5, `test_encode_with_thumb_strip_emits_expected_count`,
   `test_emitted_pixel_buffer_outlives_callback`). The mux they were blocked on
   exists; `encode_pipeline_e2e` now asserts the real thing on real containers,
   and `ThumbnailFrame` owns its pixels by construction so the lifetime stub had
   nothing left to guard.

---

## Open findings the lead should see

- **The HDR proxy preserves HDR *tagging*, not HDR *range*.** Measured: a flat
  RGB 128 through the 8-bit lane with BT.2020/PQ tags round-trips to exactly
  `128.00`; the same grey through HEVC Main10 with the same tags reads back at
  `77.67`. That gap is VideoToolbox inverse-tone-mapping display-referred BGRA
  into PQ on the encode side, and it lines up with the PQ code for the input at
  a ~50-nit diffuse-white mapping (predicted 330.9, observed ~331 — that
  agreement is also what proves the 10-bit read is right, see M8). The cause is
  structural: `to_cpu_rgb` applies the YCbCr matrix but no EOTF, so the source's
  HDR range is already gone before the encoder sees it. **Real HDR fidelity
  needs a 10-bit path through `VideoFrame::to_cpu_rgb` / `encodable_from_cpu`
  that keeps the transfer function.** Out of Track G's scope; the seam is named.
- **"Tone-map at the reader" is not implemented, only correctly *tagged*.**
  iOS gets tone-mapping free by handing `AVVideoColorPropertiesKey` to the
  `AVAssetReader`. redshift's reader has no equivalent. What Track G fixed is
  the half that was actively wrong: the SDR variant used to inherit the source's
  BT.2020/HLG tags (`VideoMuxTrack.color = stream.color`), describing normalized
  pixels with the source's colorimetry — the exact bug `203deb5d9` fixed
  upstream. The mux track now carries the plan's output colour. Same seam as
  above.
- **`p420` (`0x70343230`) is still unsupported, deliberately.** It appears only
  for the non-production combination "HEVC Main10 + BT.709 tags". Probed live:
  `planes=2, w=64 h=48, bytesPerRow plane0=256, plane1=256, widthOfPlane
  0=64 1=32` → **4 bytes per sample**, not the 2 that `x420` uses. I would have
  had to guess the layout, so it keeps erroring loudly with the format code in
  the message. A 10-bit **SDR** source could plausibly land here; measurement
  recorded so a follow-up starts from data.
- **Not mine and currently red in this tree:** `cargo deny check licenses`
  (loro's MPL-2.0/BSL-1.0 crates, ReplicaMan). Unchanged by Track G — the only
  dependency decision I made was to *not* add one.

---

## Test evidence

Run on macOS 25.5.0, cargo/clippy 1.95.0.

| Suite | Count | Result |
|---|---|---|
| `ferment-engine-core --lib` (incl. 4 new `video_rate_control_tests`) | 18 | **ok, 0 failed** |
| `ferment-engine-media` full (unit + 8 integration targets, incl. 2 new 10-bit narrowing tests) | 47 + 23 | **ok, 0 failed** |
| `ferment-engine-processors --lib` (incl. 11 new `thumb_container::tests`) | 944 | **ok, 0 failed**, 28 ignored |
| `ferment-engine-processors --test video_encoder` | 21 | **ok, 0 failed** |
| `ferment-engine-processors --test encode_pipeline_e2e` | 9 | **ok, 0 failed** |
| `cargo check -p ferment-project` | — | clean |
| `cargo fmt --all -- --check` | — | **clean workspace-wide** |
| `cargo clippy --all-targets -D warnings` on core/media/processors | — | clean on every file I touched |
| `cargo check` + `cargo clippy --target x86_64-pc-windows-msvc -p ferment-engine-media` | — | clean |

```
$ cargo test -p ferment-engine-processors --test video_encoder --test encode_pipeline_e2e

running 9 tests
test the_resolved_proxy_bit_rate_is_identical_across_independent_probes ... ok
test audio_only_sources_reencode_to_aac_m4a ... ok
test proxy_preset_emits_thumbnails_during_the_single_decode_pass ... ok
test ai_preset_transcodes_a_real_h264_fixture_and_probes_back ... ok
test a_ten_bit_hdr_output_reads_back_at_the_measured_pq_level ... ok
test videotoolbox_delivers_presentation_order_when_reordering_is_off ... ok
test one_decode_feeds_every_variant_and_the_strip ... ok
test the_sdr_variant_of_an_hdr_source_is_derived_from_the_finished_proxy ... ok
test an_hdr_source_splits_the_decode_and_tags_the_derived_variant_bt709 ... ok
test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.54s

running 21 tests
test a_multi_variant_request_needs_variants_unique_ids_and_a_thumb_handler ... ok
test a_backendless_multi_variant_encode_validates_first_then_reports_the_boundary ... ok
test a_source_with_no_stated_frame_rate_resolves_the_proxy_at_thirty ... ok
test a_thumb_strip_without_a_handler_is_refused ... ok
test both_stock_presets_state_reordering_off_through_the_settings_builders ... ok
test detect_hdr_transfer_reads_hlg_and_pq_only ... ok
test hdr_settings_tag_hlg_and_pq_as_bt2020_main10 ... ok
test planning_the_same_source_twice_resolves_the_same_rate ... ok
test portrait_four_k_proxies_at_1080_wide_not_606 ... ok
test proxy_bit_rate_is_deterministic_bits_per_pixel ... ok
test a_backendless_encoder_reports_the_boundary_and_writes_nothing ... ok
test proxy_scales_by_short_edge_and_ai_by_long_edge ... ok
test scaled_size_never_upscales_and_stays_even ... ok
test sdr_settings_state_bt709_rather_than_inheriting_source_tags ... ok
test the_ai_feed_holds_a_fixed_average_regardless_of_geometry ... ok
test the_ai_preset_states_frame_reordering_off_on_a_bounded_gop ... ok
test the_hdr_proxy_preserves_its_transfer ... ok
test the_proxy_preset_is_all_intra_with_reordering_off ... ok
test the_encode_trait_validates_the_thumb_strip_before_the_backend_boundary ... ok
test cancelling_mid_encode_delivers_neither_output_nor_partial ... ok
test the_output_path_is_invisible_until_the_encode_completes ... ok
test result: ok. 21 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

```
$ cargo test -p ferment-engine-processors --lib thumb_container
running 11 tests
test thumb_container::tests::a_thumbnail_that_is_not_bgra_or_is_short_is_refused ... ok
test thumb_container::tests::thumbnail_cells_are_baseline_jpeg_that_decodes_back_to_the_frame_size ... ok
test thumb_container::tests::an_empty_container_is_a_pckt_v2_header_plus_zeroed_slots ... ok
test thumb_container::tests::a_second_reader_sees_cells_the_writer_has_published ... ok
test thumb_container::tests::a_full_strip_refuses_the_next_cell ... ok
test thumb_container::tests::reopening_resumes_the_append_cursor_past_published_cells ... ok
test thumb_container::tests::reset_rewinds_the_strip_for_a_re_run ... ok
test thumb_container::tests::appending_publishes_bytes_then_slot_then_count ... ok
test thumb_container::tests::open_or_create_replaces_a_container_reserved_for_a_different_length ... ok
test thumb_container::tests::a_file_that_is_not_a_v2_strip_is_refused_loudly ... ok
test thumb_container::tests::a_strip_round_trips_encoded_thumbnails ... ok
test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 961 filtered out
```

```
$ cargo test -p ferment-engine-core --lib video_rate_control
running 4 tests
test capabilities::video_rate_control_tests::the_same_inputs_resolve_to_the_same_rate_every_time ... ok
test capabilities::video_rate_control_tests::an_average_rate_ignores_geometry_and_frame_rate ... ok
test capabilities::video_rate_control_tests::an_unusable_frame_rate_falls_back_to_thirty ... ok
test capabilities::video_rate_control_tests::bits_per_pixel_multiplies_output_pixels_by_frame_rate ... ok
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 14 filtered out
```

### Determinism, shown rather than claimed

- `the_same_inputs_resolve_to_the_same_rate_every_time` — 1 000 repeats of
  `BitsPerPixel(0.75).resolved_bit_rate(1080, 1920, 59.94)`, all equal, pinned
  at **93 218 688**.
- `planning_the_same_source_twice_resolves_the_same_rate` — two independent
  `plan()` calls produce `VideoCodecSettings` that compare equal, bit rate
  **93 218 688**.
- `the_resolved_proxy_bit_rate_is_identical_across_independent_probes` — two
  independent probes of the same file on disk → identical `SourceVideo` →
  identical settings, and the value is re-derived from the formula in the test
  rather than compared to a stored constant.

### Mutation testing — every behavioural test proven to fail when the code is wrong

Each mutation was applied, the suite run, and the mutation reverted.

| # | Mutation | Result |
|---|---|---|
| M1 | proxy preset `caps_short_edge: true → false` (the July long-axis bug) | **4 red**: `portrait_four_k_proxies_at_1080_wide_not_606`, `proxy_scales_by_short_edge_and_ai_by_long_edge`, `proxy_bit_rate_is_deterministic_bits_per_pixel`, `planning_the_same_source_twice…` |
| M2 | `scaled_size` always caps the long edge | **5 red** (M1's four plus `scaled_size_never_upscales_and_stays_even`) |
| M3 | fps fallback deleted (`let fps = frames_per_second`) | **1 red** in `--test video_encoder`, **1 red** in engine-core (`an_unusable_frame_rate_falls_back_to_thirty`) |
| M4 | `group_of_pictures` returns `true` for reordering when `b_frames == 0` | **3 red**: `the_ai_preset_states_frame_reordering_off_on_a_bounded_gop`, `the_proxy_preset_is_all_intra_with_reordering_off`, `both_stock_presets_state_reordering_off_through_the_settings_builders` |
| M5 | HDR settings stop asking for HDR metadata | **1 red**: `hdr_settings_tag_hlg_and_pq_as_bt2020_main10` |
| M6 | mux track takes a default colour instead of the plan's | **1 red**: `an_hdr_source_splits_the_decode_and_tags_the_derived_variant_bt709` |
| M7 | **VideoToolbox never states `AllowFrameReordering`** | **1 red**: `videotoolbox_delivers_presentation_order_when_reordering_is_off` — *"reordering-off must not offset composition times"*. This is the important one: it proves the platform default really does reorder, so the preset assertion is not the only thing holding the line. |
| M8 | ten-bit narrowing takes the high byte (the `v216` rule) | **1 red** e2e (`a_ten_bit_hdr_output_reads_back_at_the_measured_pq_level`) + **2 red** engine-media unit |
| M9 | SDR variant re-reads the source instead of the finished proxy | **1 red**: `the_sdr_variant_of_an_hdr_source_is_derived_from_the_finished_proxy` |
| M10 | `append` never writes the cell bytes | **5 red** thumb-container tests |
| M11 | index slot stride 16 → 8 | **3 red** thumb-container tests |
| M12 | `reset` stops rewinding the append cursor | **1 red**: `reset_rewinds_the_strip_for_a_re_run` |

M1 and M4 were re-run after the `VideoEncodingPlan` simplification and stayed
red with the same test names.

### Edges exercised by running, not by assertion

- The reordering property set to **both** `true` and `false` on a live
  VideoToolbox session — neither is refused, so a platform that *did* refuse
  would fail the test rather than silently reorder.
- Colour primaries/transfer/matrix set for BT.709 and BT.2020+PQ on live
  sessions — accepted for both H.264 and HEVC Main10.
- `HDRMetadataInsertionMode` on an H.264 session — **refused, -12902**, captured
  above.
- A malformed thumb strip in four shapes: wrong magic, wrong version, index
  table that disagrees with `data_offset`, truncated below the header. All four
  error with the reason named.
- A thumbnail in the wrong pixel format and a truncated one — both refused
  before touching libjpeg.
- Concurrent writer + reader on one strip through two independent handles:
  the reader observes `valid_count` go 0 → 1 and reads the published cell.

### What I could NOT run

- **Anything on Windows.** No Windows box in this session. The Media Foundation
  changes (`AVEncMPVDefaultBPictureCount`, the HDR-metadata refusal) type-check
  and lint clean for `x86_64-pc-windows-msvc` but are **unverified at runtime**.
  `encode_pipeline_e2e` is `#![cfg(target_os = "macos")]`, so on the Windows box
  it needs the same suite re-pointed at the MF backends — the file's header
  comment already says that, it is now more true than before.
- **Linux.** No `x86_64-unknown-linux-gnu` toolchain installed here; CI is the
  first Linux check. The video code is platform-gated, so on Linux
  `UnavailableVideoIo` applies and `encode_pipeline_e2e` compiles to zero tests.
  The pure suites (`video_encoder`, `thumb_container`, `video_rate_control`) are
  platform-independent and will run there.
- **A real 10-bit HDR camera source.** The only HDR fixture is
  `synthetic-hdr-nclx.mp4` — 8-bit HEVC carrying BT.2020/PQ nclx tags. It
  exercises the tagging, the split, the derivation and the 10-bit *read-back*
  (via our own Main10 proxy), but not a genuinely 10-bit *input*. If the iOS
  `hdr-hlg-12s.mov` fixture becomes obtainable, point the split test at it.

---

## What the lead needs for wiring

**Preset names** are unchanged: `"proxy"` and `"ai"`, still returned by
`VideoEncoder::proxy_preset()` / `ai_preset()`. `crates/project`'s
`VideoEncodeVariant::{Proxy, Ai}` mapping needs no edit.

**Capability signature for the `videoVariants` consolidation:**

```rust
let encoder = VideoEncoder::new().with_backends(backends);
let result: MultiVideoEncodingResult = encoder.encode_variants(
    &source_path,
    &[
        VideoEncodingVariant { id: "proxy".into(), output_path: proxy_path,
                               preset: VideoEncoder::proxy_preset(),
                               thumb_strip: with_thumb_strip.then(|| ThumbStripOptions {
                                   interval: 0.5, max_width: 192, max_height: 192 }) },
        VideoEncodingVariant { id: "ai".into(), output_path: ai_path,
                               preset: VideoEncoder::ai_preset(), thumb_strip: None },
    ],
    progress,
    Some(&sink),   // &dyn VariantThumbnailSink
).await?;
// result.variants["proxy"] / ["ai"] -> VideoEncodingResult { output_path, bytes,
//     width, height, preset, wall_seconds, thumbnails_emitted }
```

`encode_variants` refuses an empty request, duplicate ids, and a strip without a
handler — all as `EngineError::InvalidConfiguration`, before any file is
touched. The HDR split is internal; callers pass one flat list.

**Thumbstrip wiring** (this is what unblocks `crates/project/src/runtime.rs:69`
and the `.thumbstrip` artifact suffix):

```rust
// once, before the encode — iOS sizes it ceil(duration/interval)+1, 2400 when
// the duration is unknown
let max_count = if duration > 0.0 { ((duration / 0.5).ceil() as u32 + 1).max(1) } else { 2400 };
let mut strip = ThumbContainer::open_or_create(&strip_path, max_count)?;
strip.reset()?;                       // first cell of THIS encode truncates
// per emitted cell, on one writer thread
strip.append(&encode_thumbnail_jpeg(&thumbnail)?)?;
```

Readers open their own `ThumbContainer` and poll `valid_count()` / `cell(i)`;
the format is byte-compatible with the iOS `PCKT` v2 reader.

**Types the lead now has:** `VideoRateControl`, `VideoEncodingVariant`,
`MultiVideoEncodingResult`, `VariantThumbnailSink`, `MultiVideoEncoding`
(engine-core); `SourceVideo`, `VideoCodecSettings`, `ThumbContainer`,
`IndexEntry`, `encode_thumbnail_jpeg`, `THUMBNAIL_JPEG_QUALITY`
(engine-processors). `VideoEncodingPlan` and `VideoColorProperties` are **gone**
— `plan()` returns `VideoCodecSettings` and colour is `ferment_engine_media::ColorInfo`.

**Gates:** no Justfile change is needed. `just engine-oracle` line 140 already
runs `cargo test -p ferment-engine-processors --features ort`, which covers both
new suites, and CI's `cargo test --workspace` covers them too. No new fixtures
were added, so `CANARY_ENTRIES` is untouched.

**Files changed:**
`crates/engine-core/src/capabilities.rs`;
`crates/engine-media/src/{io.rs,frame.rs,platform/macos.rs,platform/windows_media_foundation.rs}`,
`crates/engine-media/tests/{videotoolbox.rs,mux_platform.rs}` (constructor
update only);
`crates/engine-processors/src/{video_encoder.rs,encode_pipeline.rs,lib.rs}`,
`crates/engine-processors/tests/{video_encoder.rs,encode_pipeline_e2e.rs}`.
**New:** `crates/engine-processors/src/thumb_container.rs`.
