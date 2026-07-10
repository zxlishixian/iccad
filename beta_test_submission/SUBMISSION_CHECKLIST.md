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
- [x] Every routing branch has a process watchdog and validates exact header/row count before accepting output.
- [x] Runtime class maps case count and reference `k` independently to official scales, uses the larger scale, then applies a metadata-only context-line estimate.
- [x] Q&A provides no 10M/100M tier marker; uncertain/extended inputs remain capped at the conservative 100-second Final limit.
- [x] Context estimation does not open or parse trace logs.
- [x] Missing, failed, slow, malformed, or incompatible embedding access falls back to deterministic inference.
- [x] Forced failures of multi-view, calibrated-dual, deterministic, and large k-means branches all produce a valid CSV; the final fallback is singleton buckets.
- [x] Small-input budget is 18 seconds primary plus 7 seconds fallback, targeting the original 30-second Final limit.
- [x] Runtime inference does not read gold/golden/meta/trace.
- [x] No API configuration or credential is packaged.
- [x] Public set1/set2 pass through the top-level executable.
- [x] Canonical five-view route passes a fresh-cache 160-case Final-time stress test in 70.13 seconds.
- [x] Forced timeout and unavailable-binary injections produce valid output for 7-, 37-, 240-, and 640-case routes.

## User actions before upload

- [ ] Confirm the organizer's Beta upload destination and deadline.
- [ ] Upload the *contents* directly into the organizer-provided `beta_test_submission` folder; do not add an extra directory or archive layer unless explicitly requested.
- [ ] After upload/download, verify that `regr_fail_bucketing` remains executable.
- [ ] Keep an immutable local copy of the submitted folder.
- [ ] Recheck the final router SHA-256 below after transfer.

## Final hashes

- Router SHA-256: `fc7ed6b79a17058c3ec03074d6b8511fdc5274ee4d888762e1353a7d409df963`.
- Canonical multiview SHA-256: `f69eeae39db888f37a54dcd51f376e21cb340d1c8a7b257f0105d5ecc750779f`.
