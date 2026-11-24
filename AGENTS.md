# Ferment Perception Engine

This is a public repository. The canonical upstream is `lockvoid/ferment-perception-engine`.

- Read [README.md](README.md) and the relevant design notes before changing a processor or runtime contract. Keep editor, project and installer behavior in the consuming application.
- Check the source and redistribution terms of every imported file. Preserve its license and notices. Keep credentials, local paths, private data, model weights and external test media out of commits. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- Keep libav dynamically linked and the engine FFmpeg build LGPL. Select inference backends through the `coreml` and `ort` features. Keep the `--no-default-features --features ort` build working.
- Prove each fix with a test that fails before the change and passes afterward. Run the affected suite and `just check`. Set `ORT_DYLIB_PATH` before ORT tests. Run the relevant oracle when model or golden behavior changes and report skipped cases.
- Stage explicit files. Commit only at the owner's request. Write concise English commit messages without co-author trailers.
