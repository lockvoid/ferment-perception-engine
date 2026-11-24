# Windows ML automatic provider qualification · 2026-11-16

The owner explicitly replaced the old DirectML-first Windows requirement with
Windows ML catalog discovery, installation, registration and automatic device
selection. `OrtPlacement::WindowsMl` now describes the 14 previously GPU-qualified
shipping artifacts. Other families retain their CPU placement. Android retains
its explicit CPU/XNNPACK overrides. Existing artifact bytes and hashes are unchanged.

The native boundary uses the pinned Microsoft.Windows.AI.MachineLearning 2.5.96
package: ORT 1.28.3, catalog DLL and built-in DirectML. Catalog-certified EPs are
prepared on a blocking model-load worker and registered into the same ORT env.
`MaxPerformance` selects the device; no NVIDIA or DirectML provider is hard-coded
in the default chain. Failed preparation/session creation falls back to explicit
CPU. A CPU placement explicitly selects CPU devices so a process-wide catalog
registration cannot override it through ORT's default MaxEfficiency policy.

On Windows Server 2025 build 26100, RTX A4000, driver 621.14, the catalog offered
NVIDIA TensorRT RTX 2.30.51.0. Download happened through Windows ML; SSH lacked a
logon session, requiring server-only Appx registration of the staged package.
The shipping implementation has no PowerShell installer workaround. A clean
first install in an interactive Windows 11 profile remains a distinct untested
installation scenario.

## Native results

All rows used Windows ML automatic selection, with the real native per-node
assignment checked after session construction. All weighted nodes in these runs
were assigned to NVIDIA; none fell back to CPU.

| Models | Evidence |
|---|---|
| GLAP audio/text | 19/6 inputs; embedding cosine, max error and norm gates; 671/995 NVIDIA nodes |
| SigLIP image/text | 7/13 inputs; original goldens; 536/491 NVIDIA nodes |
| MERT 95M/330M | 14 inputs and 28 outputs each; original per-output PSNR gates; 607/1128 NVIDIA nodes |
| BeatThis 1/2 | 70/70 beats, worst delta 0 s; 18/18 downbeats; 1079 nodes each |
| Autoshot | identical 11 scene boundaries; 116 nodes |
| RealESR general/WDN/anime | worst u8 error 2/5/1; 70/70/38 nodes |
| FastDVDnet | worst u8 error 1; changing padded dimensions and 4×4 lower boundary; 145 nodes |
| RIFE | natural-frame PSNR 53.99–56.94 dB; unaligned dimensions; 321 nodes |

The independent public Rust retrieval test also passed original PCM/RGB/text,
RU/EN rankings, repeated inference, eviction/reload and explicit CPU placement
AFTER NVIDIA registration (1 test, 155.93 s). Enhancement passed 5 tests, 56.59 s.
The enhancement GPU bound is the already approved provisional ±12 u8 from the
lead audit at the top of `016-report.md`, not a new tolerance. The initial run
accidentally used the CPU-calibrated ±3 and rejected WDN's measured error of 5;
that failed log is retained. RIFE retains the approved 45 dB floor.

## Defects reproduced and corrected

- `ort-sys` rc.12 omits `KernelInfo_GetOperatorSinceVersion` from its API table.
  Reading graph assignment called CreateEnvWithOptions instead. Bindgen now
  derives assignment functions from the unmodified pinned API-28 headers and
  accesses the same DLL/session. Registration/device APIs precede the mismatch.
- TensorRT RTX's implicit dynamic profile uses opt=1. GLAP audio compilation
  produced zero patch tokens; FastDVDnet compilation failed shape constraints
  and silently ran on CPU. Explicit profiles describe GLAP's 4..1001 mel frames
  and FastDVDnet's dimensions divisible by four. The latter retains the EP's
  implicit i16 ceiling rounded to 32764, with opt=256 and actual-shape allocation.
  The public frontend, model bytes, padding and arithmetic were not changed.

Compact logs, runtime hashes, all 73 numerical inputs / 101 outputs, and actual
assignments are in Hypr's `docs/dev/evidence/windows-ml-20261116/`.

## Final application and code checks

The real Windows application passed four analysis jobs and 24 expected top-1
searches through Python/TypeScript Live SDK clients, before and after reopening
a fresh isolated profile. Its 38 assignment log records show NVIDIA TensorRT RTX
for all four GLAP/SigLIP encoders, with no CPU EP. Evidence, EXE/DLL hashes and
executed scripts are in `app-final/`. This used local copies of the unchanged
model artifacts; it is not CDN publication or interactive GUI acceptance.

Windows all-targets Clippy for the native facade/infer/models/retrieval passed.
Mac portable compilation, targeted Clippy and 144 engine unit tests passed;
Hypr analysis passed 49 tests. The configured full nextest run had 1627 passes,
one existing VFR packet-duration failure, and 329 skips. Full `just check` also
encounters unrelated processor lint/format changes, and the full oracle lacks
AmberNet artifacts. These workspace gates are not reported as green.

The complete application/SDK report, installation limits and QA cleanup are in
`../ferment-hypr/docs/dev/windows-ml-20261116.md`. No commit or model republishing
was performed. This report does not certify all Windows functionality, the first
clean interactive Windows 11 installation, or other GPU vendors.
