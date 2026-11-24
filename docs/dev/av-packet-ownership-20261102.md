# Interleaved packet ownership regression

Scoped engine-media bug found by Ferment Hypr's G6 actual AV stress. No applicable
AGENTS.md exists in this repository. Read libav input/decode/audio/packets and
the actual Hypr caller before changing code. Native frame/audio/packet APIs,
codec selection, permissions, source checks and formats must not change.

Executed caller regression (Hypr commit `8a3f9e6a`, branch `ux/redesign`):

    cargo test -p ferment-hypr --bin ferment-hypr --offline distinct_movie_evictions -- --ignored --test-threads=1 --nocapture

Run from the sibling ferment-hypr workspace, sharing its existing build target.
Actual checked MOV input, 16 unique movies plus a still, paced test output/no OS
device. After warm-up 64 visual frames force 1088 checked opens and RSS grows
467776→999360 KiB; a 128 MiB growth gate fails (6.34 s). Log:
`/tmp/ferment-distinct-decoder-retirement-20261102.log`.

macOS leaks on a disposable debug-entitled copy of that test reports 180314
leaks/618464224 bytes. The root is AVBufferRef allocation while decoding;
10 KiB payloads match the interleaved PCM packets. Original app/target binary
and user projects are not signed or modified. Report:
`/tmp/ferment-decoder-disposal-leaks-debug-20261102.log`.

The video feed declares one Packet outside its stream-filtering loop, then
overwrites it on skipped audio packets without unref. The audio feed and raw
PacketReader have the same shape. Intended repair: scope each owned Packet to
one loop iteration so existing RAII drops skipped packets as well as selected,
EOF and error paths. No manual background cleanup or cache-budget increase.

Required GREEN: same actual Hypr regression, checked AV playback/PCM/pixels,
engine-media packet/audio/decode tests and native leak report repeat. Other
working-tree files remain untouched. Recovery is a forward revert of these
three packet-lifetime edits only. This RED record does not claim the fix exists.

## GREEN

Moved each owned Packet into one iteration of the video feed, audio feed and
PacketReader loops. Skipped packets now drop through the existing owner before
the next av_read_frame; selected/EOF/error paths retain the same behavior.
No unsafe code, codec option, timing, API, grant or memory budget was added.

Unchanged Hypr target PASS 6.16 s: 1088 actual checked opens, RSS
380432→385696 KiB (+5.1 MiB instead of +519 MiB). Log:
`/tmp/ferment-distinct-decoder-retirement-green-20261102.log`.
Wider engine-media and live/cold AV regression, leak-scanner repeat and isolated
Dev/Release profile remain explicit follow-up gates, not claimed by this test.

## Wider repeat and lint cleanup

The fixed disposable debug-entitled Hypr test copy now reports **0 leaks /0
total leaked bytes** under macOS `leaks --atExit` (peak footprint 310.0 MiB).
`/tmp/ferment-decoder-disposal-leaks-fixed-20261102.log`. The original target
binary and user app are unchanged. The former leaking copy reported 180314
leaks /618464224 bytes.

Engine-media full library repeat: **97 PASS /2 existing opt-in**, 0.78 s.
Strict `cargo clippy --lib -- -D warnings`: **PASS**. Nine constant-sized
chunk iterators in frame conversion/probe use `as_chunks`, preserving existing
channel/alpha/stride behavior. Two comments now correctly state that
avcodec_send_packet retains needed buffer references while the caller continues
to own/drop its Packet. No production behavior beyond the three RAII loop fixes
is changed. Logs `/tmp/ferment-engine-media-lint-regressions-20261102.log` and
`/tmp/ferment-engine-media-packet-clippy-green-20261102.log`.

Commands are run from ferment-hypr with its shared `CARGO_TARGET_DIR` and this
repository's `--manifest-path`, avoiding duplicate ordinary build caches.
Hypr's actual SDK suite also passes 17 scenarios (140.82 s); the isolated live
repeat passes 3 tests (26.23 s). Its strengthened RSS measurement runs one
fixed priming window then two fixed measured windows in a new process, with
the original 128 MiB gate: both measured 1088-open windows remain at 385920 KiB.
That harness change is not a decoder budget increase or memory trimming.
