# ICCAD 2026 Problem B Beta Submission

## Required interface

```bash
./regr_fail_bucketing --input <input.csv> --output <output.csv> --k <k>
```

The top-level executable is a POSIX shell router. Every backend is a self-contained Linux x86_64 PyInstaller application. No Python, pip, GPU, Docker, or package installation is required. Successful execution writes exactly `Case,bucket`, with one row per input case.

## Final-time-aware routing

The router uses the original Final runtime limits as its safety target, rather than the relaxed Beta limits. The official interface exposes no benchmark number or `Max Lines` tier. `B_QA_20260612.pdf` confirms only that Alpha/Beta enlarge the outer timeout; it does not define a tier environment variable or extra CLI argument. Therefore the router never assumes that it owns the 300-second tier.

The router first maps both observable size signals onto the official scale:

| Scale | Max cases | Reference `k` |
|---:|---:|---:|
| 1 | 10 | 2 |
| 2 | 30 | 4 |
| 3/7 | 100 | 8 |
| 4/8 | 300 | 16 |
| 5/9 | 1000 | 32 |
| 6/10 | 3000 | 64 |

Case count and soft `k` are mapped independently, then the larger scale wins. Thus an undersized `k` cannot hide a large input, while a large `k` conservatively upgrades a small input. The resolved scale is then combined with a metadata-only context-line estimate:

| Observable class | Conservative Final cap | Primary / deterministic retry |
|---|---:|---:|
| resolved scale has max 10/30 cases, context is 1M-like | 30 s | 18 s / 7 s |
| resolved scale has max 10/30 cases, context looks larger | 30 s | 15 s / 10 s |
| resolved scale has max 100–3000 cases, context is 1M/10M-like | 100 s | 85 s / 10 s |
| resolved scale has max 100–3000 cases, context is 100M-like | 100 s | 75 s / 15 s |

The last row may correspond to an official 100M/300-second benchmark, but remains capped at 100 seconds because the tier cannot be identified safely. Reserving more deterministic-retry time is safer than risking an external 100-second kill after assuming 300 seconds.

The line estimate uses only file sizes under the input directory: ordinary log bytes are divided by 128 and compressed-log bytes by 24, calibrated from the public data. It does not open or parse trace logs. The estimate controls only how early recovery starts and can never increase the watchdog above 100 seconds.

Within those budgets, `n <= 160` uses the canonical five-view model, `161 <= n <= 300` uses the calibrated dual-input backend, `301 <= n <= 900` uses deterministic agglomerative clustering, and larger inputs use deterministic k-means. Missing embeddings go directly to deterministic inference. Timeout, malformed embeddings, incompatible dimensions, invalid header/row count, or backend failure triggers the deterministic retry; if that also fails, the router emits one singleton bucket per case.

Every attempted backend first removes stale output. A result is accepted only if it has the exact `Case,bucket` header and one row per input case. The environment variables `BETA_ONE_M_CONTEXT_EST_LINES`, `BETA_LONG_CONTEXT_EST_LINES`, `BETA_MULTIVIEW_MAX_CASES`, `BETA_MULTIVIEW_WALL_TIMEOUT`, `BETA_FULL_MAX_CASES`, `BETA_FULL_WALL_TIMEOUT`, and `BETA_AGGLOM_MAX_CASES` are available only for controlled validation overrides.

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
- Router SHA-256: `13528c31ffe65f55bc19f1a837c085a02a12c89e2a3ec8694dee6e0b3553e026`.

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
