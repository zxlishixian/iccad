# Beta v2 Baseline-First Validation

> Historical record (superseded 2026-07-13): the timing table below describes
> the earlier size-routed candidate. The current feedback build attempts the
> canonical five-view backend at every official case/`k` scale while retaining
> the same baseline-first watchdog. See `VALIDATION_RESULTS.md` and
> `README_SUBMISSION.md` for the active policy and hashes.

Validation date: 2026-07-12.

This candidate preserves the frozen Beta models and binaries. Only the router
policy changes:

1. Atomically publish a singleton emergency output.
2. Run the deterministic backend and atomically publish a validated baseline.
3. Run the stronger embedding backend in a separate candidate file.
4. Atomically replace the baseline only when the expert exits successfully and
   its output has the exact `Case,bucket` shape.

The router uses the original Final limits (30 or 100 seconds), not the relaxed
Beta limits. Expert failure, timeout, missing configuration, malformed output,
or dimension mismatch preserves the already-published baseline.

## Cold-cache comparison

The same seven datasets, local 768-dimensional OpenAI-compatible embedding
endpoint, and Final-limit harness were used for both routers.

| Dataset | Cases | Current Beta | Beta v2 | v2 margin | Final route |
|---|---:|---:|---:|---:|---|
| first_batch | 80 | 43.12 s | 40.36 s | 59.61 s | five-view |
| stage2 | 240 | 88.19 s | 72.81 s | 27.13 s | baseline after expert timeout |
| stage3 | 640 | 7.93 s | 4.98 s | 94.98 s | baseline only |
| VCS | 40 | 15.49 s | 19.11 s | 80.84 s | five-view |
| directed | 37 | 18.18 s | 20.43 s | 79.55 s | five-view |
| public set1 | 7 | 13.41 s | 14.10 s | 15.86 s | five-view |
| public set2 | 25 | 17.67 s | 20.95 s | 9.05 s | five-view |

The public set2 candidate was rerun with the packaged 22-second public expert
budget. It completed in 20.95 seconds and produced an output byte-identical to
the frozen Beta five-view result. All seven v2 runs produced valid outputs.

## Failure injection

- Expert forced to one second: deterministic baseline preserved, exit 0.
- Missing `LLM_MODEL_CONFIG`: deterministic baseline preserved in 1.73 seconds.
- Missing deterministic binary and missing embedding configuration: valid
  singleton output preserved, exit 0.
- Invalid baseline and expert outputs: singleton output preserved in unit tests.

Eleven unit tests covering the v2 router, generic anytime wrapper, and
selective multi-view utilities passed.

## Integrity

- The original `alpha_test_submission` and `beta_test_submission` directories
  were not modified.
- No gold, golden, meta, trace, or Python source files are included.
- No symbolic links are included.
- Top-level entry and backend executables have executable permissions.
- The frozen full and multi-view backend hashes are unchanged.
