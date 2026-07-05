# Alpha Submission Checklist

## Verified locally

- [x] Target folder is named `alpha_test_submission`.
- [x] No extra compression layer is used.
- [x] Required executable is exactly `regr_fail_bucketing`.
- [x] Executable mode is `755`.
- [x] Interface is `--input <input.csv> --output <output.csv> --k <k>`.
- [x] Output columns are exactly `Case,bucket`.
- [x] Package is self-contained; no pip/requirements/environment install is needed.
- [x] Linux x86_64 executable; packaged ELF maximum is GLIBC 2.28.
- [x] Package contains no symlinks.
- [x] `LLM_MODEL_CONFIG` is parsed as YAML content, not a path.
- [x] Only the embedding API is called; completion is not called.
- [x] Missing/failed LLM access falls back to deterministic no-trace inference.
- [x] Public benchmark runtimes are below the 90-second Alpha limits.
- [x] Formal prediction does not read gold/golden/meta/trace.
- [x] No API keys or credentials are included.

## User actions before upload

- [ ] Confirm the organizer still accepts the upload. The provided announcement states
      **June 12, 2026, 17:00 GMT+8**, which is before the current date.
- [ ] Upload the contents directly into the organizer-provided Google Drive folder
      named `alpha_test_submission`; do not rename or add a zip/tar layer.
- [ ] After upload/download, verify the executable bit or organizer-side launch behavior.
- [ ] Verify SHA-256 of `regr_fail_bucketing`:
      `8772045d823720981a07d78413c6edced55877e074422c45391353c1084a91cd`.
- [ ] Keep a local copy because late correction/re-submission may not be accepted.
