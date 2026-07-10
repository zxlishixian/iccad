# Beta Submission Checklist

## Verified locally

- [x] Folder is named `beta_test_submission`.
- [x] Required executable is at the folder top level and named `regr_fail_bucketing`.
- [x] Top-level and multi-view executable modes are 755.
- [x] Interface is `--input <input.csv> --output <output.csv> --k <k>`.
- [x] Output columns are exactly `Case,bucket` with one row per input case.
- [x] Package is self-contained; no Python, pip, requirements installation, GPU, or Docker is needed.
- [x] Linux x86_64 package; 826 ELF files require at most GLIBC 2.28.
- [x] Package contains no symlinks.
- [x] `LLM_MODEL_CONFIG` is parsed as YAML content, not a path.
- [x] Only embedding APIs are called; completion is not called.
- [x] Missing, failed, or slow embedding access falls back to deterministic inference and writes valid output.
- [x] Runtime inference does not read gold/golden/meta/trace.
- [x] No API configuration or credential is packaged.
- [x] Public set1/set2 pass through the top-level executable.
- [x] Canonical five-view route passes a fresh-cache 160-case Final-time stress test in 70.13 seconds.
- [x] Forced timeout fallback produces a valid 160-case output in 3.88 seconds.

## User actions before upload

- [ ] Confirm the organizer's Beta upload destination and deadline.
- [ ] Upload the *contents* directly into the organizer-provided `beta_test_submission` folder; do not add an extra directory or archive layer unless explicitly requested.
- [ ] After upload/download, verify that `regr_fail_bucketing` remains executable.
- [ ] Keep an immutable local copy of the submitted folder.
- [ ] Recheck the final router SHA-256 below after transfer.

## Final hashes

- Router SHA-256: `3ba0024ce225a37f4cf04b761ccf1cb4067a5775241e0f61d4364772e555a134`.
- Canonical multiview SHA-256: `f69eeae39db888f37a54dcd51f376e21cb340d1c8a7b257f0105d5ecc750779f`.
