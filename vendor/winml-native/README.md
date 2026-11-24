# Windows ML native boundary

Synchronous, safe facade for the public `WinMLEpCatalog.h` C API from
Microsoft.Windows.AI.MachineLearning 2.5.96 and ORT API 24. The caller must run
`prepare` on a blocking worker: Windows may download/install certified EPs.
It uses the catalog DLL beside the **already loaded** ORT, registers into
`ort::environment::Environment::current`, and retains registrations for the
process lifetime. It never chooses a vendor or installs CUDA itself.

Catalog/registration errors are returned. Individual download failures are
reported alongside available devices; callers can use Windows ML's built-in
providers and report that fallback. Failed providers are retried on the next
preparation; successful registrations are reused. No session is created here.
The catalog owns EP pointers, enumeration owns callback data, and all strings
are copied before their owners are released. No raw pointer crosses this API.
Graph assignments are owned snapshots of ORT's assignment information, enabled
by `session.record_ep_graph_assignment_info=1`; availability is not execution.

`include/` contains unmodified C headers from the pinned NuGet (archive SHA-256
`35d6e84feb4142e8cecaf36704fa3ab0fc8251d892850fb80e8341f7d09653a3`).
Bindgen derives the assignment API table from these headers. `ort-sys` rc.12
omits `KernelInfo_GetOperatorSinceVersion`, so its later API-24 fields have
different offsets from Windows ML 1.28.3. Registration/device APIs precede
that mismatch; assignment uses a separate, correctly generated API-28 view
of the same runtime and session. It fails explicitly with older runtimes.

The native A4000 catalog probe and real-model goldens are recorded in Hypr's
`docs/dev/windows-ml-20261116.md`. This is an internal runtime boundary shared
by UI, SDK and agent analysis, with no separate user operation or stored schema.

Sources: [catalog installation](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/initialize-execution-providers),
[registration](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/register-execution-providers),
[device policy](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/select-execution-providers).
