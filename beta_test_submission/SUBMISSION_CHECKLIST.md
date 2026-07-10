# Beta Submission Checklist

## Verified locally

- [x] Target folder is named `beta_test_submission`.
- [x] Required executable is at the folder top level and named exactly `regr_fail_bucketing`.
- [x] Executable mode is 755.
- [x] Interface is `--input <input.csv> --output <output.csv> --k <k>`.
- [x] Output columns are exactly `Case,bucket` with one row per input case.
- [x] Package is self-contained; no Python/pip/requirements installation is needed.
- [x] Linux x86_64 package; all 826 ELF files require at most GLIBC 2.28.
- [x] Package contains no symlinks.
- [x] Docker is not used.
- [x] `LLM_MODEL_CONFIG` is parsed as YAML content, not a path.
- [x] Only embedding APIs are called; completion is not called.
- [x] Missing/failed/slow LLM access falls back and still writes valid output.
- [x] Runtime prediction does not read gold/golden/meta/trace.
- [x] Package contains no API keys or credentials.
- [x] Public set1/set2 and 640/3000-case routes complete within Beta limits.
- [x] Router SHA-256 is `902515919a70120292a6dd14643e6f2a69ea283f86a5377d020e989657071e2e`.

## User actions before upload

- [ ] Confirm the organizer's Beta upload destination and deadline.
- [ ] Upload the contents directly into the organizer-provided `beta_test_submission` folder; do not add an extra directory or archive layer unless explicitly requested.
- [ ] After upload/download, verify that `regr_fail_bucketing` remains executable.
- [ ] Recheck the top-level router SHA-256 against the value above.
- [ ] Keep an immutable local copy of the submitted folder.
