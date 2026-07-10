# ICCAD 2026 Problem B Beta Submission

## Required interface

```bash
./regr_fail_bucketing --input <input.csv> --output <output.csv> --k <k>
```

The top-level executable is a POSIX shell router. Every backend is a self-contained Linux x86_64 PyInstaller application. No Python, pip, GPU, Docker, or package installation is required. Successful execution writes exactly `Case,bucket`, with one row per input case.

## Final-time-aware routing

The router uses the original Final runtime limits as its safety target, rather than the relaxed Beta limits. It validates every backend result before accepting it: the file must have the exact `Case,bucket` header and exactly one row for every input case. A stale output is removed before each backend attempt.

- `n <= 30`, embedding configuration present: canonical five-view, five-seed GBDT, with an 18-second watchdog.
- `31 <= n <= 160`, embedding configuration present: the same canonical five-view model, with an 85-second watchdog.
- `161 <= n <= 300`: calibrated dual-input backend, with an 85-second watchdog.
- Primary-model failure, timeout, malformed embedding, incompatible embedding dimension, or invalid output: deterministic Drain/SVD/agglomerative retry. Its limit is 8 seconds for `n <= 30` and 10 seconds otherwise.
- `301 <= n <= 900`: deterministic Drain/SVD/agglomerative backend with an 80-second watchdog.
- `n > 900`: deterministic Drain/SVD/k-means backend with an 80-second watchdog.
- Missing embedding configuration: deterministic no-trace backend directly.
- If the deterministic backend itself fails or yields an invalid CSV, the router emits one singleton bucket per input case. This is a valid scored submission, generally safer than a failed benchmark scoring zero.

The `18 + 7` small-input budget leaves explicit watchdog grace within the original 30-second Final limit even when the embedding process is unavailable. Medium routes reserve an 85-second primary attempt plus a 10-second deterministic retry under the original 100-second Final target. Environment variables `BETA_MULTIVIEW_MAX_CASES`, `BETA_MULTIVIEW_WALL_TIMEOUT`, `BETA_FULL_MAX_CASES`, `BETA_FULL_WALL_TIMEOUT`, and `BETA_AGGLOM_MAX_CASES` exist for controlled validation overrides. The router reads only `input.csv` to choose a route; model backends then read the referenced sim/regr logs.

## Canonical five-view model

The multi-view route reads bounded `sim.log(.gz)` and `regr.log(.gz)` samples. It constructs five views: global features, global summary, ordered event evidence, PC/opcode/register/CSR objects, and local sim/regr context. Each view is reduced to 64 dimensions using training-fitted reducers.

The model uses five frozen GBDT seeds. Each seed blends dual-view and five-view pair probabilities at `0.50/0.50`; seed probabilities are averaged and clustered with average-link agglomerative clustering using the supplied reference `k`.

For scalable inference, all logs are memoized in-process, case-index-only document prefixes are canonicalized, duplicate view documents are embedded once, and vector pair relations are built with NumPy. The artifacts were retrained with the identical canonicalization, so training and inference match. Runtime never reads `gold.csv`, `golden.csv`, `meta.csv`, or `trace.log`; trace and completion are disabled.

## LLM configuration

`LLM_MODEL_CONFIG` must contain YAML content. The package calls only the organizer-provided embedding endpoint; completion is never called. If embeddings are unavailable, malformed, slow, or return an incompatible dimension, the router first tries the deterministic backend and finally emits a valid singleton CSV if that backend is also unavailable.

## Packaging compatibility

- Target: Linux x86_64, RHEL/Alma/CentOS 8 compatible.
- PyInstaller: 6.20.0, onedir layout.
- All packaged ELF files require at most `GLIBC_2.28`.
- Symlinks: 0 after materialization.
- Top-level executable mode: 755.
- Package size: approximately 519 MiB.
- Router SHA-256: `f0c53ec586d6b46579fec922e8db84311fd8d1c6b085d189bca70b3c56974035`.

## Final validation summary

| Test | Cases | Route | BA | TPR | TNR | Wall |
|---|---:|---|---:|---:|---:|---:|
| public benchmark_set_1 | 7 | canonical five-view | 1.000000 | 1.000000 | 1.000000 | 13.00 s |
| public benchmark_set_2 | 25 | canonical five-view | 0.950469 | 0.982906 | 0.918033 | 17.42 s |
| cold 160-case stress | 160 | canonical five-view | n/a | n/a | n/a | 70.13 s |
| normal set1 recheck | 7 | canonical five-view | n/a | n/a | n/a | 2.49 s |
| forced 1-second small multi-view guard | 7 | deterministic agglomerative | n/a | n/a | n/a | 2.72 s |
| forced 1-second 241-case dual guard | 240 | deterministic agglomerative | n/a | n/a | n/a | 7.58 s |
| forced k-means large route | 640 | deterministic k-means | n/a | n/a | n/a | 5.32 s |
| disabled deterministic binary | 7 | singleton emergency output | n/a | n/a | n/a | < 1 s |

The 160-case stress input is derived from existing labeled data but uses only `input.csv` and logs during inference. The forced-failure rows deliberately use invalid watchdog/binary settings; they validate output continuity, not hidden-score quality. The measurement is a runtime guardrail, not a hidden-score estimate.

## Files

- `regr_fail_bucketing`: required timeout-aware router.
- `multiview/`: canonical five-view binary and frozen five-seed artifacts.
- `regr_fail_bucketing_full`, `_internal/`, `models/`: calibrated Alpha dual-input backend for 161–300 cases.
- `fast/`: deterministic-only large-set backend and multi-view failure fallback.
- `VALIDATION_RESULTS.md`: detailed protocol and validation record.
- `SUBMISSION_CHECKLIST.md`: upload checklist.

Python sources and `requirements.txt` are intentionally omitted from the submitted package.
