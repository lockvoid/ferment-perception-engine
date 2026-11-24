# Generated video fixtures

`generate.sh` creates the 3 container × 3 codec matrix used by the integration
tests. Every clip is a deterministic 64×48, 12-frame test pattern. H.264 and
HEVC clips use different GOP intervals and two B-frames so packet DTS, PTS,
sync-sample seeking, and presentation reordering are all exercised. ProRes is
intra-only by design.

It also creates the compressed-audio matrix: AAC-LC in M4A, MP4, MOV, and MKV,
at 44.1 kHz and 48 kHz, in mono and stereo. Each one-second signal starts with
a non-zero cosine component so priming or one-sample alignment errors are
observable in PCM parity tests.

The HE-AAC fixtures (`aac-he-*`) are encoded by Apple's `afconvert` (FFmpeg
cannot encode SBR without libfdk), so regeneration is macOS-only; the script
skips them elsewhere. They deliberately cover both real-world container
flavors — the dual-rate afconvert shape (stsd/mdhd at the 22.05 kHz AAC core
rate, no edit list) and the effective-rate FFmpeg-remux shape (stsd/mdhd at
44.1 kHz with a priming edit list) — plus HE-AACv2's implicit parametric
stereo (mono core, stereo output).

The files can be checked in or regenerated in CI:

```sh
./crates/engine-media/tests/fixtures/generate.sh
```

Generation requires a developer FFmpeg build with `libx264`, `libx265`, and
`prores_ks`. FFmpeg is never invoked by a default library build.

FFmpeg's strict ISO MP4 muxer does not accept ProRes. The `.mp4` ProRes fixture
therefore uses the QuickTime (`qt`) brand while retaining the `.mp4` extension;
it exercises the same ISO-BMFF demux path and extension-independent probing.

Conversely, FFmpeg's MOV muxer writes a QuickTime v1 AAC sample entry. Audio
`.mov` fixtures use the MP4 muxer with the QuickTime (`qt`) major brand,
preserving a real QuickTime-branded ISO-BMFF file while keeping the portable v0
`mp4a` sample entry used by the platform-adapter matrix.

AudioToolbox golden PCM is compared with the `ffmpeg-dev` decoder at a 35 dB
SNR floor. The explicit M4A priming test also requires the decoded frame count
to equal the source rate and the best first-sample alignment lag to be zero.
Native random seeks remain bit-identical to native sequential PCM.

`fetch-conformance.sh` downloads two small, official ITU-T H.265.1/JCT-VC
Main-profile vectors (AMVP and random-access/RAP) from the ITU archive and
losslessly wraps their Annex-B streams in MP4 for the platform-session tests.
Run that script separately when you need the conformance cases. The downloaded
vectors stay outside Git.

`Video/upscale-three-aac-tracks.mov` (68 114 bytes, SHA-256
`01b3d290e63ddcbb8b3f69817509939a21f1c9b2bfe32040a8501f603705077f`) is a
byte-identical copy of the Cuts iOS fixture of the same name: a three-track
AAC MOV whose delayed tracks `libav_audio_delayed.rs` checks against frozen
stream tables. A fixture root in `FERMENT_ENGINE_FIXTURES_DIR` still wins.
