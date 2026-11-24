# 2026-09-02_02 — libav as the media backend (REPORT)

Tracks `docs/fleet/2026-09-02_02_libav-media-backend.md`. One section per
stage, written as each lands. Rulings are cited by number.

---

## The pin: FFmpeg 8.1.2

### 2026-11-10 — build 5: source float recording decode

Hypr's explicit source-linear recording uses standard rawvideo/NUT rather
than labeling straight RGBA as premultiplied OpenEXR. Build5 adds only the
`rawvideo` decoder and `nut` demuxer to the existing component whitelist.
The checked MediaInput format whitelist admits NUT; protocol and secondary-I/O
restrictions stay unchanged. The source/ABI pin and dynamic LGPL2.1+ posture
stay unchanged. Native macOS artifact SHA-256:
`091fbc2d40f557e8b4037ed3d0b2925b0ccbf55be527d373ec7c37c704cd4ad7`.
Other-platform build5 artifacts and CDN publication remain release gates;
immutable build4 artifacts are not overwritten.

Native float input and explicit `CpuRgba32Float` output retain packed float
samples without BGRA8 conversion, transfer functions or alpha association.
Explicit float output bypasses automatic integer hardware surfaces. Float
resizing requires an explicit graph operation; this does not enable general
PQ/HLG decoding or certify physical HDR devices. Ordinary integer decode
defaults retain their previous behavior.

Independent CLI fixtures grade all four literal channels (negative/HDR RGB,
nonopaque alpha and hidden color at alpha0), rational30000/1001 PTS and EOF.
Both pathname/Native and checked-stream/explicit-float routes pass for31
frames and a one-frame recording. The NUT index exposes last PTS rather than
the complete presentation interval. The probe now observes bounded tail
packet PTS+duration in the native time base instead of adding a nominal-FPS
epsilon. One-stream container duration and each observed video stream retain
that exact extent, including the one-frame case whose index duration is0.
Multi-stream global interleave/offset duration remains libav's; it is not
certified by these single-stream recording tests. Tail work is capped at4096
packets/128MiB and errors are reported. Fixture provenance:
`crates/engine-media/tests/fixtures/linear-rgba32f.md`.

### 2026-11-10 — build 4: exact CFR recording timestamps

Hypr's native recording/import replay at60 and30000/1001FPS reproduced
10/31 and15/31 preceding-frame selections. Matroska's1ms timestamps do not
represent those frame boundaries. The existing encoder now writes FFV1/AVI;
build4 adds AVI to the demuxer whitelist and the checked MediaInput format
whitelist. The protocol whitelist and secondary-I/O rejection are unchanged.
Identical rational-v2 requests now replay31/31 exact RGBA frames at both rates.
Permanent decoder fixtures independently pin31 literal color/alpha frames,
integer-cross-multiplied PTS, EOF, and both file/checked-stream paths. Both
regressions pass; old Matroska/FFV1 support remains. Native macOS artifact SHA:
`3cfb0d91c2a75078de748fb61dfe7eca9daa52ae5afc7ebce4e0eefc215dbbfe`.
All10111 original source files match the pinned8.1.2 archive. The dynamic
LGPL2.1+ posture, source/ABI pin and native dependency gate are unchanged.
Other platform artifacts/CDN publication remain separate release gates.

### 2026-11-09 — build 3: saved lossless recording playback

The Ferment Hypr VF11 recording scenario found a concrete missing codec:
Matroska import succeeded, but ordinary media playback failed with
`no decoder for AV_CODEC_ID_FFV1`. Build 3 adds only the FFV1 decoder to the
shared whitelist; the source/ABI pin, dynamic LGPL posture, dependency gate
and existing platform encoder lists are unchanged. The native macOS artifact
is built from the pinned source archive and tested in the portable Hypr QA
bundle. Other platform artifacts and CDN publication are separate release
gates; build 2 is not silently overwritten under its immutable artifact name.
`libav_ffv1` retains the regression with independently encoded literal RGBA
quadrants, including nonopaque alpha.

Ruling 3 asks for 8.0.x, with the decision procedure "read
`ffmpeg-sys-next` 9.0.0's `build.rs`, decide from the source, record the pin".
Read, and measured:

`build.rs:994-1013` maps libavcodec versions to cfgs. It emits **`ffmpeg_8_0`
at lavc 62.8** and `ffmpeg_8_1` at 62.28 (also `ffmpeg_9_0` at 63.1). So an
8.x pin is supported and the ruling's fallback to 7.1.x is not needed.

Built `ffmpeg-next` 9.0.0 + `ffmpeg-sys-next` 9.0.0 against real 8.x headers
before writing any code: clean, with `ffmpeg_8_0=true` and `ffmpeg_8_1=true`
emitted. **8.1.2 rather than 8.0.x** because 8.0 is a closed branch upstream
(8.0.3 is its last release, no further security fixes) and 8.1.2 is the newest
8.x. Both satisfy the cfg test the ruling made decisive.

The pin is asserted, not assumed:
`platform::libav::tests::links_the_pinned_libavformat_major` fails on a box
that provisioned a different ABI. Observed: `libavformat 62.12.102`.

## Deviation from ruling 4: we build all three artifacts, BtbN is not used

Ruling 4 says the Windows and Linux artifacts are "the pinned BtbN
`*-lgpl-shared` releases re-hosted on our CDN". Measured their
`autobuild-2026-09-01-13-13` build on 2026-09-02, and it cannot carry this
track's own licensing posture:

| what ruling 6 states | what BtbN's `lgpl-shared` actually is |
| --- | --- |
| "LGPL-2.1-or-later" | configured `--enable-version3` → **LGPL-3**, and its `LICENSE.txt` is LGPL v3 |
| "the exact configure line" in the notice | ~40 further third-party libraries (libaom, dav1d, SVT-AV1, openh264, libass, libbluray, libjxl, librav1e, openal, SDL2, AMF, ffnvcodec …), each with its own notice and source-offer obligation |
| ruling 16: "files only; no `interrupt_callback`" | `--enable-libssh --enable-librist --enable-libsrt` — three network protocol stacks |
| — | **137 MB of DLLs** against 8.4 MB for ruling 3's whitelist |

Ruling 3 already states a Windows configure line, which only has a use if we
build Windows ourselves — rulings 3 and 4 contradict each other on this point,
and ruling 3 is the one that keeps rulings 6 and 16 true. So: **macOS builds
natively, Windows and Linux cross-build in a pinned container**, all three from
the same source tarball and the same component whitelist. Only ruling 4's
*mechanism* changes; its outcome (pinned artifacts, digest-verified, re-hosted
on our CDN) is unchanged.

Switching back is a one-function change in `scripts/provision-ffmpeg` if the
LGPL-3 posture and the 40-library notice turn out to be acceptable.

## Two corrections to ruling 3's component lists

Both were link/decode failures, not preferences:

1. **`av1_videotoolbox` added to the macOS hwaccel list.** libavcodec's own
   `av1` decoder is hwaccel-only — `ffmpeg -h decoder=av1` reports
   "Threading capabilities: none / Supported hardware devices: videotoolbox",
   and ruling 3 declines dav1d. Without the hwaccel, an AV1 file does not
   decode at all rather than decoding slowly.
2. **Both `d3d11va` spellings on Windows** (`h264_d3d11va` *and*
   `h264_d3d11va2`, likewise hevc/vp9/av1). Upstream gates the implementing
   object on the non-"2" config (`libavcodec/Makefile:1061`:
   `OBJS-$(CONFIG_H264_D3D11VA_HWACCEL) += dxva2_h264.o`), so enabling only
   the "2" variant links a decoder referencing a hwaccel nobody built. This
   was a hard `undefined reference to ff_h264_d3d11va2_hwaccel` link failure.

3. **`image2` and `wav` REMOVED from ruling 3's demuxer list, and `wav` from
   its muxer list.** Ruling 3 says to read `probe_import` and add what stills
   need. Reading it (`apps/desktop/src-tauri/src/lib.rs:712`) shows the
   opposite requirement: `probe_import` reaches its dedicated still reader
   (the `image` crate) and its WAVE reader ONLY when the container probe
   FAILS. A libavformat that opens a PNG therefore reclassifies a still import
   as `mediaKind: video`, and one that opens a WAVE renames its codec from
   `lpcm` to `pcm_s16le`. Both were caught as real `ferment-studio` test
   failures in Stage 2, and the fix would otherwise have to be in
   `src/lib.rs`, which is outside this track's Boundary. Nothing in the
   workspace reads WAVE through engine-media either — every consumer uses
   `hound` — and `MuxFormat` is Mp4 or Mov, so there is no WAVE to write. The
   `pcm_*` DECODERS stay: a PCM track inside a MOV is still decodable.

## Artifacts

Built by `just provision-ffmpeg-build`, staged in `dist/ffmpeg-publish/8.1.2/`,
awaiting `scripts/ffmpeg-publish`'s upload step (Dimitri runs `aws s3 cp`).

| target | installed | packed | sha256 |
| --- | --- | --- | --- |
| `aarch64-apple-darwin` | 8.4 MB | 3.0 MB | `b654391743152ef459a6bacbf9cf7ced96c57bebe42249c2f821a6998b4bd464` |
| `x86_64-pc-windows-msvc` | 11 MB | 3.0 MB | `e8d296aa9683c226db2611cd183157ace7b38acdb758b78bdb5aadef4f50f036` |
| `x86_64-unknown-linux-gnu` | 9.2 MB | 3.0 MB | `8c700b245ae370976a32414087ade17e060cfe5561c15c71f66948efcc99cd83` |

Served under `…/ferment/studio/vendor/ffmpeg/<version>/` (ruling 6, path),
alongside the source tarball.

Source, and the LGPL-2.1 §6 offer the notice file points at:

    https://ffmpeg.org/releases/ffmpeg-8.1.2.tar.xz
    sha256 464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c

macOS install names are `@rpath/libav*.N.dylib` (ruling 5), verified with
`otool -D`. `configure`'s own verdict is the licence gate and prints
`License: LGPL version 2.1 or later`; the script refuses any build whose
`config.mak` carries `CONFIG_GPL=yes` or `CONFIG_NONFREE=yes`.

### The dependency gate (lead review, corrected)

The first build of these artifacts linked Homebrew. All six macOS dylibs named
`/opt/homebrew/opt/libx11/lib/libX11.6.dylib`, because `configure` auto-detects
whatever it finds on the build box and this one has X11 installed. An earlier
draft of this report claimed the only non-`@rpath` dependencies were
`libSystem`, `libbz2` and `libz`; that was wrong — it was written from one
library's `otool` output, read too quickly, and libX11 was in it.

The failure mode is the reason this now has a gate rather than a habit: such an
artifact links, builds, and passes every test on the machine that made it, and
aborts in dyld on a user's machine at first import.

Fixed with `--disable-autodetect` on all three targets, plus explicit
`--enable-zlib` everywhere (Matroska track compression, MOV `cmvd`),
`--enable-bzlib` on macOS, and `zlib1g-dev` / `libz-mingw-w64-dev` added to the
cross container so the compression support is not silently lost there. Windows
and Linux were already clean — measured, not assumed.

The dependency set is now ASSERTED, in two places:

- `scripts/provision-ffmpeg` checks it after every build **and after every
  download**, because a CDN artifact can name `/opt/homebrew` just as easily as
  a local one. macOS via `otool -L`; Windows and Linux inside the same
  container that built them, since `objdump` and `readelf` for those targets
  are not on a Mac.
- `crates/engine-media/tests/provision_ffmpeg.rs` checks the vendored tree the
  dev binary actually loads, and — so the predicate cannot rot into a no-op —
  asserts that it REJECTS Homebrew's own `libavcodec`, which is the exact shape
  of the failure.

Allowed, per platform:

| target | allowed dependencies |
| --- | --- |
| macOS | `@rpath/`, `/System/Library/`, `/usr/lib/` |
| Windows | KERNEL32, msvcrt, USER32, ole32, WS2_32, Secur32, CRYPT32, ncrypt, bcrypt, d3d11/dxgi/mf\* and our own siblings |
| Linux | libc, libm, libpthread, libdl, librt, libz, libatomic and our own siblings |

d3d11 and the Media Foundation DLLs do not appear in the Windows import table:
FFmpeg resolves those with `LoadLibrary` at runtime, which is why that list is
as short as it is.

The gate then caught a SECOND artifact-only failure the same day, which is the
argument for having it: with `--enable-zlib` added, the Windows build began
importing `zlib1.dll`. Windows has no system zlib, Debian's mingw package ships
both a static archive and an import library, and configure prefers the import
library — so the artifact named a DLL that would not exist on a user's machine.
Fixed by removing the import library from the cross image, which leaves the
static archive as the only candidate; zlib now rides inside our own DLLs and
`THIRD_PARTY_NOTICES.md` credits it (zlib License, a permissive one).

Two ordinary script bugs surfaced alongside, both of the same shape — a shell
construct whose last status becomes the caller's: `cleanup()`'s
`[[ -n "$SCRATCH" ]] && rm` returned 1 when there was nothing to clean, and
`main`'s final `if` returned 1 for every non-host target. Both made a
successful provisioning run exit non-zero, which would have failed CI.

**Until the artifacts are on the CDN, nothing provisions.** They are staged and
digest-verified; `scripts/ffmpeg-publish` prints the exact upload commands and
`scripts/ffmpeg-publish --check` verifies the edge afterwards.

---

## Stage 1 — provision + link + license

No behaviour change: the libav backend is initialized and its log stream is
bridged, but nothing calls into it yet.

**Landed.** `scripts/provision-ffmpeg` (download + `--build` + `--package`),
`scripts/ffmpeg-publish`, `just provision-ffmpeg-{mac,win,linux,build}` and
`just check-windows`, `.cargo/config.toml` `FFMPEG_DIR`, `.gitignore`,
`ferment-engine-media` gains `ffmpeg-next` 9.0.0 + `log`, `deny.toml`
exceptions, `apps/desktop/src-tauri/THIRD_PARTY_NOTICES.md`, the Account/Legal
line, the rpaths, `tauri.conf.json` frameworks + resources, `ci.yml` (Linux
provision + `libclang-dev` + `zstd`, and the `windows-latest` job of ruling
13), `release.yml` provision step, `.zed/tasks.json`,
`platform/libav/{mod,log,error}.rs`.

### Three things the doc did not anticipate

**`cargo:rustc-link-arg` does not propagate.** The `@rpath` install names mean
every binary that links libav needs an rpath of its own, and cargo has no
directive that carries a link argument from a dependency to a dependent — a
`ferment-engine-processors` test binary aborted in dyld before `main`. Rather
than five copies of the same twenty lines, this is `crates/ffmpeg-rpath`, a
build-dependency whose `emit()` each consumer's `build.rs` calls in one line:
`engine-media`, `engine-processors`, `project`, `sequencer-audio`,
`sequencer-video`, and the Tauri crate (which also emits ruling 5's
`@executable_path/../Frameworks`).

**The Windows cross-check of ruling 13 needs a different triple.** Ruling 13
proposes `cargo check --target x86_64-pc-windows-msvc`. Bindgen needs a libc
for the target, and no Mac has the Windows SDK; mingw's headers describe the
GNU ABI and clang refuses to parse them under an MSVC target. `just
check-windows` therefore checks **`x86_64-pc-windows-gnu`** against mingw
headers extracted from a container, while parsing the MSVC artifact's own
FFmpeg headers. It proves what ruling 13 wants — the `cfg(target_os =
"windows")` code compiles — and takes 1m50s. `ferment-project` is out of it:
it pulls `ring` through rustls, whose C sources cc-rs compiles for real. The
authoritative Windows gate is the CI job, which runs the tests.

Aside: bookworm's mingw-w64 (10.x) lacks `d3d12video.h`, which FFmpeg's public
`hwcontext_d3d12va.h` includes unconditionally; the sysroot comes from
`debian:trixie-slim` (mingw-w64 12.0) for that reason.

**Ruling 6 puts the notice at `apps/desktop/THIRD_PARTY_NOTICES.md`, the
Boundary at `apps/desktop/src-tauri/`.** It is at the latter, because
`bundle.resources` paths resolve from the Tauri crate directory. The file also
carries a short ONNX Runtime (MIT) section: it is named "third-party notices"
and ships in the bundle, and omitting the other bundled native library would
make it untrue.

### Gates (macOS, aarch64-apple-darwin)

| gate | result |
| --- | --- |
| `cargo clippy --all-targets -- -D warnings` | 0 |
| `cargo fmt --all -- --check` | 0 |
| `cargo deny check licenses` | `licenses ok` |
| `cargo test -p ferment-engine-media -p ferment-engine-processors -p ferment-project` | pass (52 / 941 / 8 + integration) |
| `cargo test -p ferment-sequencer-video -p ferment-studio` | pass |
| `just check-windows` | pass |
| `pnpm check` | 0 errors, 0 warnings (1253 files) |
| `pnpm test:unit` | 888 passed, 106 files |

`crates/engine-media/tests/provision_ffmpeg.rs` exercises the download path
end to end: it packages the libraries this very test binary is linked against
into a `file://` mirror, provisions from it into a scratch directory
(`FERMENT_FFMPEG_VENDOR`, so the live vendor tree is never touched), and then
tampers with the sidecar to prove a digest mismatch is refused. It caught two
real script bugs — a `trap … RETURN` that leaked out of its function and
failed a run that had already succeeded, and a cross-build that inherited
Mach-O objects from the host build ("file format not recognized").

One pre-existing lint fixed to make the gate green on a Mac:
`platform/macos.rs:1347` `useless use of vec!`, in a macOS-only test module
that Linux CI never compiles.

### Still owed on Stage 1

- The three artifacts are staged but **not published**; CI cannot go green
  until they are on the CDN (`scripts/ffmpeg-publish`).
- The `windows-latest` CI job's URL, once it has run on Dimitri's push.

---

## Stage 2 — probe + the parity table

`platform/libav/probe.rs` implements ruling 7; `demux::probe` forwards to it
and every consumer sees the same `MediaInfo` shape. `ContainerFormat` gained
`Other(String)` and lost `Copy`, and ruling 1 predicted the consumer diff
exactly: one `_` arm in `demux/mod.rs`, one `&` on `normalize_packet_data`'s
container argument, one `#[derive(Copy)]` dropped from a test struct. Nothing
else in the workspace moved.

### The parity table

`crates/engine-media/tests/libav_parity.rs`. **48 fixtures compared, 3 refused
by both, 196 tolerated differences, 0 untolerated.** The test prints every
tolerated difference it hits with its reason, so the itemized list is the test
output rather than a copy in this file.

Three comparisons are equivalences rather than tolerances, because the two
sides say the same thing in different words:

- **time bases and frame rates compare as rationals** (ruling 12 says so):
  Matroska's `1000000/1000000000` IS libav's `1/1000`.
- **an absent display matrix equals an identity one.** libavformat emits the
  side data only when there is a transform; the old demuxer always returned
  the `tkhd` matrix. A real rotation still has to match exactly, and
  `synthetic-rotation.mov` proves it does.
- **durations compare within 1 ms**, in seconds, so a time-base change alone
  is not a difference.

Difference paths carry the stream KIND — `streams[0:video].sample_rate` — so a
tolerance can say "the old demuxer filled an audio field on a video track"
without also excusing that field where it means something.

The tolerated differences fall into eleven classes, every one of them a case
where libav is right and the hand-built demuxer was wrong or silent. The
sharpest, checked against `ffprobe` on each fixture:

| class | what changed |
| --- | --- |
| `sample_rate`/`channels` on video | the old demuxer reported 64 Hz and 0 channels for video tracks |
| `frame_rate` on audio | it synthesized 11025/256 from AAC packet cadence |
| `duration` | libav honours the edit list. `h264-gop4.mp4` is 12 frames at 12 fps: libav and ffprobe say 1.000 s, the old demuxer said 1.083 s. `synthetic-fragmented.mp4`'s fMP4 `mvhd` duration is 0, reported verbatim before and 1.0 s now |
| `audio_timing` | libav's numbers ADD UP: `aac-lc-44100-mono.mp4` has 44 AAC frames = 45056 coded samples, and libav says 1024 + 43076 + 956 = 45056 where the old demuxer said 1024 + 43076 + 0 = 44100 |
| `language` | the old demuxer mis-decoded packed ISO-639 and returned `"\u{7f}\u{7f}\u{7f}"` and `` "```" `` for real files |
| `frame_rate` on video (3 fixtures) | one entry each, all confirmed against the files' real presentation timestamps |

### Three things the probe has to do that `codecpar` does not offer

**Priming does not live in `codecpar`.** `initial_padding` is zero for every
AAC and MP3 file in the corpus. The numbers arrive as
`AV_PKT_DATA_SKIP_SAMPLES` on the stream's FIRST packet, so the probe reads
one packet per audio stream (capped at 256) to get them. Without this, every
AAC track lost its 1024- or 2112-sample encoder delay.

**The end trim rides the LAST packet**, which a head scan cannot see and a
scan to EOF would make every probe proportional to file length. The probe
seeks to one second before the end and reads out from there — bounded work,
and the final packet is necessarily inside that window.

**libavformat drops a MOV edit list's tail trim entirely** (it keeps the head
as skip-samples side data, and `ffprobe -show_packets` confirms no
`discard_padding` is emitted for `aac-he-44100-stereo.mp4`). That remainder is
recovered arithmetically — `nb_frames × frame_length − leading − audible` —
and refused unless it comes to less than one coded frame, which is what
padding IS. This reproduces the old demuxer exactly on the round-trip test
(691 samples) and on the fixture the old demuxer got right (5 samples).
`codecpar->frame_size` is the frame length when set, but it is filled by
DECODING a frame, so it stays 0 for a stream libavcodec could not decode; the
first packet's own duration is the fallback.

### Two contracts that changed, both stated in a test

**HE-AAC `sample_rate` is now the SBR output rate.** The container flavour
used to leak through: afconvert's `.m4a` said 22050, an FFmpeg remux of the
same audio said 44100, and a consumer had to know to ask
`aac_effective_sample_rate()` instead. libavformat reports what the decoder
produces, so both now say 44100. `tests/audiotoolbox.rs`'s
`decodes_he_aac_fixtures_at_the_sbr_effective_rate` was rewritten to pin the
new contract (its core-rate-to-effective-rate scaling factor is now 1 and is
gone). This is a Stage-5 rewrite that Stage 2 forced early, because
`AudioToolboxAudioIo::probe` goes through `demux::probe`.

**`audible_samples` is `None` when libavformat estimated the duration.** An
untagged CBR MP3 has no declared length — libavformat divides file size by bit
rate, and `duration_estimation_method` says `AVFMT_DURATION_FROM_BITRATE`.
Reporting that estimate as `audible_samples` (46078 against a true 46080) made
`decodes_mp3_fixture_matrix_to_contiguous_pcm` take its exact-equality branch
and fail by two samples. `AudioTiming::audible_samples` means "after applying
container timing metadata", and there is none, so it is `None` — which is what
the old demuxer said. LAME-tagged files are `FROM_STREAM` and keep their exact
value.

### Gates

`cargo test` green for `ferment-engine-media`, `-processors`, `-project`,
`-sequencer-video`, `ferment-studio`; `clippy --all-targets -D warnings` 0;
`fmt --check` 0; `deny check licenses` ok; `just check-windows` 0; `pnpm
check` 0.

---

## STOP conditions

### 1. Two Matroska regression fixtures libav refuses that the old demuxer opened

Ruling 12's corpus comparison surfaced this, and the track's STOP list names it
("a fixture libav refuses to open that `demux::probe` opened — show both
outputs"). Recorded as a `PENDING` tolerance so the suite is green and the
question stays visible.

    regression-mkv-min-timestamp.mkv   (1.0 KB)
    regression-mkv-seekhead-cycle.mkv  (2.6 KB)

libav, both: `avformat_open_input failed: Input/output error (libav -5)`.
`ffprobe` on the same files says why:

    [matroska,webm] 0x00 at pos 52 (0x34) invalid as first byte of an EBML number
    [matroska,webm] Element at 0x67 ending at 0x9f4dbb8c53abf4 exceeds
                    containing master element ending at 0x1413

The old demuxer opened them. Both are hand-corrupted fixtures built to pin the
OLD demuxer's behaviour on malformed EBML — a seekhead cycle and a minimum
timestamp — and the corpus already carries `corrupt-truncated.mkv` and
`corrupt-bitflipped.mkv`, which BOTH sides refuse. FFmpeg's own CLI fails on
these two as well.

**Recommendation: accept the refusal and retire both fixtures in Stage 6**,
when the demuxer whose crash they pinned is deleted. Refusing a structurally
invalid file is the correct behaviour, and no code will remain for them to
regression-test. Say otherwise and the alternative is a libavformat option to
tolerate broken EBML, which would weaken every Matroska import to keep two
synthetic files openable.

---

## Stage 3 — video decode

Landed against amendments A1-A3 (2026-09-02 evening), which the measurements
below prompted and which supersede rulings 8 and 9 for macOS. A3 leaves it to
the executor whether the macOS libav build keeps its `*_videotoolbox` encoders
and hwaccels: **they stay**. They are unused now that macOS decodes and
encodes through our own sessions, but they are the fallback if a VideoToolbox
session ever fails to open, and the packed artifact is 3.0 MB either way.

### The finding that reshaped this stage

Ruling 15 makes `bench_decode_on_env_clip ≤ 1.0 s` an acceptance criterion.
The straightforward libav decode path missed it by a wide margin, and the
reason is in FFmpeg's source rather than in our wrapper —
`libavcodec/videotoolbox.c:758`:

    status = VTDecompressionSessionDecodeFrame(...);
    if (status == noErr)
        status = VTDecompressionSessionWaitForAsynchronousFrames(...);

It waits after **every frame**. That serializes a decoder built to pipeline.
Confirmed independently with the `ffmpeg` CLI (1.70 s with `-hwaccel
videotoolbox` against 0.18 s in software on the same clip), so it is not
something our loop or the `ffmpeg-next` wrapper causes.

`libavcodec/nvdec.c` has no such wait — `cuvidDecodePicture` at end-of-frame,
with the map deferred to retrieval. **This is a VideoToolbox-specific defect,
not a libav one**, which is what makes the resolution a one-platform carve-out
rather than a retreat.

### Measurements

19.4 s / 1080×1350 / 30 fps / 582 frames, release profile, 3 runs, median.

| decode path | full size | → 192×192 |
| --- | --- | --- |
| VideoToolbox session, packets from `DemuxReader` (the old code) | 0.57 s | 1.39 s |
| **VideoToolbox session, packets from libavformat (now)** | **0.49 s** | — |
| libav hardware (videotoolbox hwaccel) | 1.58 s | 2.35 s |
| **libav software + swscale (now, for scaled output)** | 0.62 s | **0.30 s** |

Both of ruling 15's decode criteria pass: **0.49 s** full decode and
**0.30 s** for the colour pass's decode+scale, against a ≤ 1.0 s budget that
the code being replaced missed at 1.39 s.

### What landed, and why it is shaped this way

**libavformat feeds every decoder.** `platform::libav::PacketReader` hands
`EncodedPacket`s to whatever wants them, and `VideoToolboxDecoder` now takes
its packets from there instead of `DemuxReader`. This is the point of the
whole track and it is preserved intact — the vendored demuxers have no
remaining consumer and still die in stage 6. It also cost nothing: 0.49 s
against the 0.57 s the hand-written demuxer managed.

**One backend type, choosing internally** (A1). `platform_video_backend()`
returns `LibavVideoIo` on every platform. On macOS it routes full-size decode
to `VideoToolboxDecoder` and encode to `VideoToolboxEncoder`, and scaled
output to libavcodec's software decoder; elsewhere everything is libav's.
Callers see one type and no `cfg`.

**macOS keeps its VideoToolbox sessions**, for the measured reason above. That
is roughly 600 lines of `platform/macos.rs`, not the whole file: the demuxers,
the muxer and the audio sessions are unaffected and still scheduled for the
funeral. Per A1 the grep-zero list loses `VideoToolboxDecoder`,
`VideoToolboxEncoder` and `VideoToolboxVideoIo`, and `objc2-core-media` stays.

**Windows takes the libav path, with NVDEC.** Ruling 3 specified `d3d11va`;
the shipping target is NVIDIA, so the build now also enables
`--enable-ffnvcodec --enable-nvdec --enable-nvenc` and the `*_nvdec` hwaccels,
with D3D11VA kept as the answer on an Intel or AMD machine. `HwDevice` tries
CUDA first and falls back.

**The LGPL posture is unchanged, and the artifact proves it rather than the
configure line claiming it.** `grep nonfree` over FFmpeg's configure returns
nothing for nvenc, nvdec, cuda or cuvid — they ask only for `ffnvcodec` (MIT
headers, pinned and digest-checked at `n12.2.72.0`) and
`dlopen`/`LoadLibrary`. In the built DLL:

    compiled in   h264_nvdec hevc_nvdec vp9_nvdec av1_nvdec       (hwaccels)
                  h264_cuvid hevc_cuvid vp9_cuvid av1_cuvid       (decoders)
                  h264_nvenc hevc_nvenc av1_nvenc                 (encoders)
                  h264_mf hevc_mf aac_mf prores_ks                (fallback)
    imports       KERNEL32.dll msvcrt.dll ole32.dll avutil-60 swresample-6
    dlopen'd      nvcuda.dll nvcuvid.dll nvEncodeAPI64.dll
    licence       License: LGPL version 2.1 or later

No NVIDIA library appears in the import table, so none is linked and none
ships. The dependency gate passes unchanged.

Getting there cost three builds, and the failure worth recording is the second:
FFmpeg's configure looks for a pkg-config named with the cross prefix and
**silently disables every pkg-config check** when it is absent, so nvdec was
refused for "not all dependencies are satisfied: ffnvcodec" while
`ffnvcodec.pc` sat in the prefix. The image now carries an
`x86_64-w64-mingw32-pkg-config` wrapper that sets `PKG_CONFIG_LIBDIR` — the
cross prefix and nothing else, so a host library can never satisfy a Windows
build.

**Unmeasured**: there is no NVIDIA hardware on this machine, so the claim that
NVDEC is FAST rests on `libavcodec/nvdec.c`'s structure, not a number. It
needs a run on the Windows box; the `windows-latest` CI job proves only that
it builds and the tests pass.

**Scaled output goes to the software decoder.** `DecodeOptions::output_size`
means "this exact size, ready for a CPU read", and its only caller
(`analyze_video_color`) reads every frame immediately. A hardware decode would
scale on the GPU and then read back — two transfers to reach a place the
software decoder writes to directly. This is a deviation from ruling 8, which
puts `output_size` through `transfer_scaled` on macOS; it is 4.5× faster and
it is what brings the colour pass inside ruling 15's budget.

`FERMENT_LIBAV_SOFTWARE=1` forces the software path, which is how the
comparison above was made and how the fallback is exercised.

### Tests

`crates/engine-media/tests/libav_decode.rs`, the rewrite of
`tests/videotoolbox.rs`'s decode half: the fixture matrix including the last
GOP (12 frames, not 11 — that frame only appears if the decoder is drained),
byte-deterministic seeks against a sequential decode in a deliberately
unordered walk, a seek never landing before its target, the exact
`output_size` contract, and a refusal when asked to open an audio stream as
video.

Plus `platform::libav::packets`' own tests: one stream's packets in demux
order with monotonic DTS, and a seek that rewinds to a keyframe at or before
its target.

**A note on a threshold that turned out to belong to libavcodec, not to
VideoToolbox.** Measured on 2026-09-02, a 64×48 H.264 clip decoded through
libavcodec's hwaccel comes back `AV_PIX_FMT_YUV420P` where a 96×64 one comes
back `AV_PIX_FMT_VIDEOTOOLBOX`: the hwaccel's `get_format` negotiation does
not offer the hardware format at that size. Our own decompression session has
no such threshold and decodes the 64×48 corpus to `CVPixelBuffer`s.

That distinction only became visible after A1 moved full-size macOS decode
back to our session — before it, `macos_frames_stay_on_the_gpu` had to be
gated on `FERMENT_BENCH_CLIP` because no fixture could prove the GPU path.
It now runs against the whole corpus, and the software path has a test of its
own (`a_scaled_request_decodes_in_software_and_lands_on_the_cpu`) that goes
through the route a user actually hits — the colour pass takes it on every
import.

---

## Stage 4 — video encode + mux

**The muxer** (`platform/libav/mux.rs`, ruling 11) replaces 2 300 lines of
hand-written ISO-BMFF with libavformat's `mov` muxer. A thin state machine
over one `AVFormatContext`: tracks declared, header written lazily on the
first packet (libavformat requires every stream to exist first), trailer on
`finalize`.

`tests/libav_mux.rs`, 10 tests. The centrepiece is a REMUX — every packet of a
real fixture read out through `PacketReader` and written back — because it
exercises the codec description, the timestamps, the interleave and the
trailer at once, with the source file as the expected answer. Also: colour
tags surviving as `colr`, both container flavours, `fast_start` proven by the
byte order of `moov` against `mdat` (walking the box tree, not searching for
the tag — `moov` appears inside `mdat` payloads by chance), and AAC priming
surviving as an edit list.

**A bug the remux caught, which would have shipped silently.** ProRes is one
codec id with six fourccs, and the container carries the fourcc. Nothing was
setting `codec_tag`, so libavformat picked its default and every ProRes proxy
came back relabelled — `apcn` written out, `apch` read back. Fixed by stating
the tag from the profile already in `Codec::ProRes`.

**ProRes in MP4 is now refused** rather than written. The hand-written muxer
allowed it — the corpus still carries a `prores-standard.mp4` from those days
— but no other tool accepts the result. Nothing in the pipeline asks for it:
the proxy preset, the only one reaching for ProRes, is `VideoFileType::Mov`.
It has a test of its own so the refusal is a decision rather than a surprise.

**Three defects the sequencer's own tests caught, after the suites above were
already green.** Worth listing, because each was invisible to the media tests
and only a consumer exercised the path:

* **`Drop` aborted the process.** A session dropped WITHOUT `finalize` — which
  is what asking a mux job for its track descriptors does — closed a dynamic
  buffer with `avio_closep`. The two sinks close differently, and using the
  wrong one is not a leak, it is a `SIGABRT`. It killed the whole
  `sequencer-video` test binary and hid two further failures under it.
* **The container duration counted padding as program.** It was derived from
  the packets' coded extent, so an audio track's encoder priming and padding
  were included: a 1/30 s clip reported 0.033375 s, because its 1604-sample
  packet wraps 1600 audible samples in 2+2 of padding. `VideoMuxTrack::
  duration` and `AudioTiming::audible_samples` both document themselves as the
  authority for exactly this, and both were ignored. Each track is now asked
  what it declared, and packet spans are only the fallback.
* **The duration rounded.** Measuring the packet extent in the STREAM's time
  base was wrong because the `mov` muxer picks its own timescale for a video
  track and rewrites the field — 1/30 s came back as 267/8000, off by 42
  microseconds. Measured in the packet's own base now.

**An in-memory session** (`LibavMuxer::memory`, `avio_open_dyn_buf`) because
`MuxOutput::bytes` is in the trait and the sequencer's export path uses it.
Ruling 11 asked who reads `bytes` before deciding: nobody in production, two
tests in `sequencer-video`. It refuses `fast_start` rather than silently
dropping it — that second pass reopens the output for reading, which a dynamic
buffer cannot do, and an export that quietly lost faststart is discovered by a
player, not by us.

**The encoder** (`platform/libav/encode.rs`, ruling 9) is the Windows path:
NVENC first, Media Foundation behind it, `prores_ks` for ProRes. It is not
`cfg`-gated to Windows, which is the point — on a Mac it runs against
`h264_videotoolbox`, so the Windows encode path is tested rather than merely
cross-compiled.

**A second bug, in my own first draft.** I hardcoded
`Packetization::LengthPrefixed` on encoded packets, on the strength of ruling
9's statement that `AV_CODEC_FLAG_GLOBAL_HEADER` guarantees avcC. It does not,
universally: `h264_videotoolbox` through libavcodec returns Annex-B start
codes with the flag set, and its packets match. Packets would have declared a
form they were not in. The packetization is now read off the encoder's own
extradata, and a test asserts the configuration and the packets agree — plus
one that the output muxes either way, which is what makes the tolerance safe
rather than merely honest.

`encode_pipeline_e2e.rs` is green against the libav muxer: 9 tests including
the HDR PQ level probe-back, the `ftyp` brands per preset, and presentation
order with reordering off.

---

## Stage 5 — audio

`platform/libav/audio.rs` (ruling 10). Decode is libavcodec's; encode is
`aac_at` on macOS, `aac_mf` on Windows, libavcodec's own `aac` where neither
loads. `tests/libav_audio.rs`, 6 tests.

Three bugs, all found by the tests and all of the same family — audio that
sounds right and is wrong:

**Matroska frames were not contiguous.** A caller stitching PCM relies on
frame N ending where frame N+1 starts. Matroska is timed in milliseconds, so
converting each frame's timestamp to samples rounds, and consecutive frames
stopped meeting. The container's timestamps now set the ORIGIN and nothing
else; after that the sample count is authoritative, because samples cannot
drift — samples are the unit.

**Seeks returned different audio than playback.** `av_seek_frame` plus a flush
lands at a packet boundary, and a compressed format's first frame after a
flush is not yet right: AAC's window overlaps its predecessor, MP3's polyphase
filter carries state. The seek now lands two coded frames early (or
`seek_preroll` where the codec declares one) and decodes into the target.

**Encoder padding was counted as program.** The last coded frame is
zero-padded to `frame_size`, and that padding was inflating
`audible_samples` — 48128 for a 48000-sample push. What the caller pushed and
what was submitted are now counted separately.

**The trailing trim was never applied.** libavcodec consumes
`AV_PKT_DATA_SKIP_SAMPLES` and drops an audio stream's LEADING priming
itself, which is easy to mistake for "libavcodec handles the trims". It does
not handle the end: the MOV demuxer emits no end discard (confirmed in stage
2), so the encoder's trailing padding arrived as real samples — 48128 where
the container declares 48000. The decoder now truncates to the declared
audible length, which ruling 10 asks for and which the AudioToolbox path did.
Found by `engine-processors`' waveform oracle, not by the audio suite: my own
test had a one-coded-frame slack that hid it. It asserts exact equality now.

**The resampler was built too early.** It was configured from the codec
context at open time, but several decoders — AAC among them — do not settle
`sample_fmt` until they have decoded something, so it could be reading a
format the frames were not in. It is built from the first frame now. This was
not the cause of the failure above; it was found while looking for it, and is
a real defect regardless.

The seek test compares numerically rather than bit-exactly, which ruling 12
allows for audio: libavcodec's AAC decoder converges on slightly different
floats depending on how much pre-roll preceded a frame. The observed
difference is ~3e-6 — about -110 dBFS, an order of magnitude below 16-bit
quantization — and the 1e-4 tolerance is still three orders tighter than a
seek that landed on the wrong frame, which would differ by O(0.1).

---

## Stage 6 — the funeral

### Grep-zero proof

Every name in ruling 14, as amended by A1, across `crates/` and `apps/`:

| name | hits |
| --- | --- |
| `re_mp4` | 0 |
| `matroska_demuxer` | 0 (see below) |
| `DemuxReader` | 0 |
| `RustMp4Muxer` | 0 |
| `Mp4MuxSession` | 0 |
| `AudioToolboxAudioIo` | 0 |
| `AudioToolboxEncoder` | 0 |
| `MediaFoundationVideoIo` | 0 |
| `MediaFoundationAudioIo` | 0 |
| `WindowsMediaBuffer` | 0 |
| `FfmpegDevVideoIo` | 0 |
| `FfmpegDevAudioIo` | 0 |
| `ffmpeg-dev` | 0 |

One residual `matroska_demuxer` string exists as a TEST NAME —
`import_probe_accepts_wave_files_outside_the_mp4_matroska_demuxer` in
`apps/desktop/src-tauri/src/lib.rs`. It is not a code reference, and that file
is outside this track's Boundary, so it stands.

A1 keeps three names off the list, and they are present as it requires:
`VideoToolboxDecoder`, `VideoToolboxEncoder`, `VideoToolboxVideoIo`.

`cargo tree -p ferment-engine-media`: no `re_mp4`, no `matroska-demuxer`, no
`objc2-audio-toolbox`, no `objc2-core-audio-types`. `objc2-core-media` remains,
per A1.

### What went

Both vendored demuxers (1.6 MB), the mp3 demuxer, `macos_audio.rs`,
`windows_media_foundation.rs`, `windows.rs`, `ffmpeg_dev.rs` and its feature,
`FrameBuffer::Windows`, and seven test files whose subjects died and whose
coverage moved to the `libav_*` suites. **`mux.rs` went from 2 314 lines to
163** — the trait types, and the one Annex-B converter the libav muxer uses.
`demux/mod.rs` went from 1 581 lines to 15: the `probe` function, at its old
name and path, so no consumer had to change.

### Two judgment calls

**`tests/libav_parity.rs` was deleted.** Its entire purpose was comparing
libav against `DemuxReader`; with `DemuxReader` gone there is nothing on the
other side. It was a migration instrument and it did its job — 48 fixtures,
196 tolerated differences, 0 untolerated, all recorded above.

**The fuzz targets were rewritten rather than deleted.** They fuzzed
`DemuxReader`. libavformat is fuzzed far harder upstream than we could manage,
but the mapping on top of it is ours: `fuzz_targets/probe.rs` now exercises
the `AVFormatContext` walk, the codec and colour translation, the timestamp
arithmetic, and `DecoderConfiguration`'s parsing of whatever bytes a file
claims are an avcC record. Same property as before — a malformed file is an
`Err`, never a panic and never a hang.

### A consumer test that had to change, and why it is not a weakening

`engine-processors`' `waveform_matches_wav_for_aac_in_m4a_mp4_mov_and_mkv`
compares the same one-second tone across four containers. After the trailing
trim landed, MP4/M4A/MOV decoded to 48000 samples and Matroska to 48128 — and
the test failed.

The temptation was to make Matroska match. Checking instead: `ffprobe
-show_packets` on that fixture shows one leading skip of 1024 samples and **no
`discard_padding` at all**. So 48128 IS the correct decode for it; nothing in
that container declares those 128 samples as padding. MP4 carries it in the
edit list, Matroska here carries nothing.

Running the same test at `HEAD` in a worktree — rather than reasoning about
deleted code — showed all four containers used to land on 99 envelope windows,
which is what made it clear the divergence was a declaration difference and
not an audio error.

The waveform extractor normalizes, so an unequal length changes every value
and turns a 2.7 ms declaration difference into a total mismatch. The test now
compares the four containers over their SHORTEST COMMON LENGTH, which is what
the assertion was always about: that the four files hold the same audio.

### Consumers that moved

`sequencer-video`'s export (`LibavMuxer`, including the in-memory session),
`sequencer-audio`'s and `engine-processors`' audio backends, and
`preview_cache.rs`'s `container_duration_ms` — which ruling 7 expected to
compile unchanged, and which needed one line because `DemuxReader::open` was
its entry point rather than `probe`.
