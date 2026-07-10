# ICCAD 2026 Problem B Beta Submission

## Required interface

```bash
./regr_fail_bucketing --input <input.csv> --output <output.csv> --k <k>
```

The top-level executable is a POSIX shell router. Every backend is a self-contained Linux x86_64 PyInstaller application. No Python, pip, GPU, Docker, or package installation is required. Successful execution writes exactly `Case,bucket`, with one row per input case.

## Final-time-aware routing

The router uses the original Final runtime limits as its safety target, rather than the relaxed Beta limits.

- `n <= 64`, embedding configuration present: canonical five-view, five-seed GBDT; 24-second guard.
- `65 <= n <= 160`, embedding configuration present: the same canonical five-view model; 90-second guard.
- Multi-view timeout or failure: deterministic Drain/SVD/agglomerative fallback, without a second embedding attempt.
- `161 <= n <= 300`: calibrated Alpha dual-input backend.
- `301 <= n <= 900`: deterministic Drain/SVD/agglomerative backend.
- `n > 900`: deterministic Drain/SVD/k-means backend.
- Missing embedding configuration: deterministic no-trace backend.

The environment variables `BETA_MULTIVIEW_MAX_CASES`, `BETA_MULTIVIEW_WALL_TIMEOUT`, `BETA_FULL_MAX_CASES`, and `BETA_AGGLOM_MAX_CASES` can override these defaults. The router only reads `input.csv` to count cases.

## Canonical five-view model

The multi-view route reads bounded `sim.log(.gz)` and `regr.log(.gz)` samples. It constructs five views: global features, global summary, ordered event evidence, PC/opcode/register/CSR objects, and local sim/regr context. Each view is reduced to 64 dimensions using training-fitted reducers.

The model uses five frozen GBDT seeds. Each seed blends dual-view and five-view pair probabilities at `0.50/0.50`; seed probabilities are averaged and clustered with average-link agglomerative clustering using the supplied reference `k`.

For scalable inference, all logs are memoized in-process, case-index-only document prefixes are canonicalized, duplicate view documents are embedded once, and vector pair relations are built with NumPy. The artifacts were retrained with the identical canonicalization, so training and inference match. Runtime never reads `gold.csv`, `golden.csv`, `meta.csv`, or `trace.log`; trace and completion are disabled.

## LLM configuration

`LLM_MODEL_CONFIG` must contain YAML content. The package calls only the organizer-provided embedding endpoint; completion is never called. If embeddings are unavailable, malformed, slow, or return an incompatible dimension, the router writes a valid deterministic result instead of failing the benchmark.

## Packaging compatibility

- Target: Linux x86_64, RHEL/Alma/CentOS 8 compatible.
- PyInstaller: 6.20.0, onedir layout.
- All packaged ELF files require at most `GLIBC_2.28`.
- Symlinks: 0 after materialization.
- Top-level executable mode: 755.
- Package size: approximately 519 MiB.
- Router SHA-256: `3ba0024ce225a37f4cf04b761ccf1cb4067a5775241e0f61d4364772e555a134`.

## Final validation summary

| Test | Cases | Route | BA | TPR | TNR | Wall |
|---|---:|---|---:|---:|---:|---:|
| public benchmark_set_1 | 7 | canonical five-view | 1.000000 | 1.000000 | 1.000000 | 13.00 s |
| public benchmark_set_2 | 25 | canonical five-view | 0.950469 | 0.982906 | 0.918033 | 17.42 s |
| cold 160-case stress | 160 | canonical five-view | n/a | n/a | n/a | 70.13 s |
| forced 1-second guard | 160 | deterministic fallback | n/a | n/a | n/a | 3.88 s |

The 160-case stress input is derived from existing labeled data but uses only `input.csv` and logs during inference. The measurement is a runtime guardrail, not a hidden-score estimate.

## Files

- `regr_fail_bucketing`: required timeout-aware router.
- `multiview/`: canonical five-view binary and frozen five-seed artifacts.
- `regr_fail_bucketing_full`, `_internal/`, `models/`: calibrated Alpha dual-input backend for 161–300 cases.
- `fast/`: deterministic-only large-set backend and multi-view failure fallback.
- `VALIDATION_RESULTS.md`: detailed protocol and validation record.
- `SUBMISSION_CHECKLIST.md`: upload checklist.

Python sources and `requirements.txt` are intentionally omitted from the submitted package.
