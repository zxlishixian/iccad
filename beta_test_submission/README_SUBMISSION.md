# ICCAD 2026 Problem B Beta Submission

## Required interface

```bash
./regr_fail_bucketing --input <input.csv> --output <output.csv> --k <k>
```

The top-level executable is a timeout-aware POSIX shell router. All backends are self-contained Linux x86_64 PyInstaller onedir applications. Evaluation does not require Python, pip, package installation, GPU, Docker, or ordinary internet access. Every successful route writes exactly `Case,bucket` with one row per input case.

## Runtime routing

- `n <= 64`, embedding configuration available: five-seed multi-view GBDT.
- Multi-view failure or 45-second wall timeout: calibrated Alpha dual-input fallback.
- `64 < n <= 300`: calibrated Alpha dual-input model.
- `300 < n <= 900`: deterministic Drain/SVD/agglomerative backend.
- `n > 900`: deterministic Drain/SVD/k-means backend.

Thresholds can be overridden with `BETA_MULTIVIEW_MAX_CASES`, `BETA_MULTIVIEW_WALL_TIMEOUT`, `BETA_FULL_MAX_CASES`, and `BETA_AGGLOM_MAX_CASES`. The router reads only the input CSV to count cases.

## Multi-view model

The new small-set route reads bounded samples from referenced `sim.log(.gz)` and `regr.log(.gz)` files and constructs five embedding views: global features, global summary, ordered event evidence, PC/opcode/register/CSR objects, and local sim/regr signal context. Each view is reduced to 64 dimensions with training-fitted reducers.

Five frozen GBDT seeds predict dual-view and five-view pair probabilities. The branches are blended at `0.50/0.50`, averaged across seeds, and clustered with average-linkage agglomerative clustering using the supplied reference `k`.

Training builds every benchmark as an independent episode and never constructs pairs across datasets. Runtime inference does not discover or read `gold.csv`, `golden.csv`, `meta.csv`, or `trace.log`. Trace and completion are disabled.

## LLM configuration

The binaries parse YAML content from `LLM_MODEL_CONFIG` and call only the organizer-provided embedding endpoint. Completion is not called. Missing configuration routes directly to deterministic no-trace inference; API errors, incompatible dimensions, missing artifacts, or timeout fall back to the calibrated dual model and then its deterministic no-trace fallback.

## Packaging compatibility

- Target: Linux x86_64, RHEL/Alma/CentOS 8 compatible.
- PyInstaller: 6.20.0, onedir layout.
- Packaged ELF files scanned: 826.
- Highest required GLIBC symbol: `GLIBC_2.28`.
- Symlinks: 0 after materialization.
- Top-level executable mode: 755.
- Package size: approximately 520 MiB.
- Router SHA-256: `e8cb6336522288effeba8315f241ea5afe1c4cbde48c75575aaed915b9f1d9e3`.

## Public validation

| Dataset | Cases | k | BA | TPR | TNR | Wall |
|---|---:|---:|---:|---:|---:|---:|
| benchmark_set_1 | 7 | 2 | 1.000000 | 1.000000 | 1.000000 | 11.78 s |
| benchmark_set_2 | 25 | 4 | 0.950469 | 0.982906 | 0.918033 | 17.88 s |
| old fake stage3 | 640 | 32 | 0.778823 | 0.847862 | 0.709783 | 5.97 s |
| repeated stage3 stress | 3000 | 64 | n/a | n/a | n/a | 19.48 s |

The 3000-case input repeats stage3 rows with absolute paths and is a runtime-only stress test.

## Files

- `regr_fail_bucketing`: required timeout-aware router.
- `multiview/`: five-seed multi-view binary and frozen artifacts.
- `regr_fail_bucketing_full`, `_internal/`, `models/`: calibrated Alpha fallback.
- `fast/`: deterministic-only large-set backend.
- `VALIDATION_RESULTS.md`: detailed validation record.
- `SUBMISSION_CHECKLIST.md`: upload checklist.

Python sources and `requirements.txt` are intentionally omitted because the official path is executable-only.
