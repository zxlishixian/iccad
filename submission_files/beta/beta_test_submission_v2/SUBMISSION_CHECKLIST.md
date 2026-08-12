# Beta Submission Checklist

## Verified locally

- [x] Local candidate is `beta_test_submission_v2`; its contents are ready to be copied directly into the organizer-provided `beta_test_submission` folder without an extra wrapper layer.
- [x] Required executable is at the folder top level and named `regr_fail_bucketing`.
- [x] Top-level and multi-view executable modes are 755.
- [x] Interface is `--input <input.csv> --output <output.csv> --k <k>`.
- [x] Output columns are exactly `Case,bucket` with one row per input case.
- [x] Package is self-contained; no Python, pip, requirements installation, GPU, or Docker is needed.
- [x] Linux x86_64 package; 827 ELF files require at most GLIBC 2.28.
- [x] Package contains no symlinks.
- [x] `LLM_MODEL_CONFIG` is parsed as YAML content, not a path.
- [x] Only embedding APIs are called; completion is not called.
- [x] Every routing branch has a process watchdog and validates exact header/row count before accepting output.
- [x] Runtime class maps case count and reference `k` independently to official scales, uses the larger scale, then applies a metadata-only context-line estimate.
- [x] Small scales retain a 7-second baseline and 18-second five-view budget under the 30-second Final cap.
- [x] Q&A provides no 10M/100M tier marker; uncertain/extended inputs remain capped at the conservative 100-second Final limit.
- [x] Context estimation does not open or parse trace logs.
- [x] Missing, failed, slow, malformed, or incompatible embedding access falls back to deterministic inference.
- [x] Forced failures of multi-view, calibrated-dual, deterministic, and large k-means branches all produce a valid CSV; the final fallback is singleton buckets.
- [x] Every official 10/30/100/300/1000/3000-case scale attempts the canonical five-view backend after publishing a fallback.
- [x] Runtime inference does not read gold/golden/meta/trace.
- [x] No API configuration or credential is packaged.
- [x] Public set1/set2 pass through the top-level executable.
- [x] Mock-backend tests confirm 300/1000/3000-case scales select canonical five-view.
- [x] Real set1 package smoke selects canonical five-view, returns a valid 8-line CSV, and completes in 3.93 seconds.
- [x] Expert watchdog terminates the complete child process group and preserves the published baseline.
- [x] Forced timeout and unavailable-binary injections produce valid output for 7-, 37-, 240-, and 640-case routes.

## User actions before upload

- [ ] Confirm the organizer's Beta upload destination and deadline.
- [ ] Upload the *contents* directly into the organizer-provided `beta_test_submission` folder; do not add an extra directory or archive layer unless explicitly requested.
- [ ] After upload/download, verify that `regr_fail_bucketing` remains executable.
- [ ] Keep an immutable local copy of the submitted folder.
- [ ] Recheck the final router SHA-256 below after transfer.

## Final hashes

- Entry SHA-256: `f6fadf592fb3ff2468d846c5b6d5c37a1bd10da62da0420ffdadb40c9992f64d`.
- Router policy SHA-256: `27c009f33d02b6681eb6cc126051d9cb74f683dcfff1c911bd020d803e592dae`.
- Sparse multiview SHA-256: `a3ecec3e411e29d21b377d45da873a84d2fcf5c474973e3a28cb7d7f05854ff9`.
- Canonical multiview SHA-256: `f69eeae39db888f37a54dcd51f376e21cb340d1c8a7b257f0105d5ecc750779f`.
