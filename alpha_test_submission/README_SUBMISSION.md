# ICCAD 2026 Problem B Alpha Submission

## Required interface

```bash
./regr_fail_bucketing --input <input.csv> --output <output.csv> --k <k>
```

`regr_fail_bucketing` is an x86_64 PyInstaller onedir executable. Its runtime is
contained in `_internal/`; evaluation does not require Python, pip, internet
package installation, GPU, Docker, or `requirements.txt`. The output has exactly
`Case,bucket` columns and one row per input case.

## Submitted model

The primary no-trace route is:

1. Read bounded samples from `sim.log(.gz)` and `regr.log(.gz)` referenced by `input.csv`.
2. Build deterministic SVD features and `features`/`summary` LLM embeddings.
3. Run the 10-seed calibrated dual-input blend:
   - rich residual focal MLP;
   - logistic/GBDT/shallow-MLP soft voting;
   - rich/ensemble alpha `0.88/0.12`;
   - rich temperature `1.15`; ensemble temperature `1.00`.
4. Blend the frozen official-style root-cause logistic adapter at alpha `0.50`.
5. Use average-linkage agglomerative clustering with the supplied soft-hint `k`.

Frozen MLP checkpoints were exported to framework-independent NPZ files. Runtime
inference uses a numerically equivalent NumPy implementation, so PyTorch and GPU
libraries are not part of the executable. The maximum measured logit difference
against the original PyTorch models is `9.54e-7`.

Prediction does not read `gold.csv`, `golden.csv`, `meta.csv`, or `trace.log`.
The trace helper modules are packaged only because the no-trace adapter reuses
anchor parsing on selected sim/regr text; no trace path is opened.

## LLM configuration

The executable reads YAML text from `LLM_MODEL_CONFIG` using the contest format.
Only the `embedding` endpoint is called; completion is not called. API errors,
missing configuration, incompatible embedding dimensions, or missing artifacts
fall back to the deterministic no-trace pipeline and still write a valid CSV.

The evaluator provides the permitted HTTP LLM endpoints. Ordinary internet access
is not required. Network latency counts toward the benchmark runtime.

## Packaging compatibility

- Target: Linux x86_64.
- PyInstaller: 6.20.0, onedir layout.
- Highest GLIBC symbol found across 287 packaged ELF files: `GLIBC_2.28`.
- No symlinks remain in the submission directory.
- Executable SHA-256:
  `8772045d823720981a07d78413c6edced55877e074422c45391353c1084a91cd`.
- Logical upload size after source/checkpoint cleanup: approximately 360 MiB.

Python sources and `requirements.txt` are intentionally omitted. The official
evaluation path is the self-contained executable.

## Public validation

| Dataset | Cases | k | BA | TPR | TNR | Final binary wall time |
|---|---:|---:|---:|---:|---:|---:|
| benchmark_set_1 | 7 | 2 | 0.722222 | 0.777778 | 0.666667 | 2.55 s |
| benchmark_set_2 | 25 | 4 | 0.921255 | 0.957265 | 0.885246 | 6.33 s |

Both outputs are byte-identical to the original PyTorch inference package.
With `LLM_MODEL_CONFIG` removed, the self-contained deterministic fallback completed
benchmark_set_1 in 2.44 seconds and emitted a valid 8-line CSV.

## Files

- `regr_fail_bucketing`: required self-contained executable.
- `_internal/`: bundled Python/native runtime.
- `models/`: frozen NPZ MLP, sklearn, reducer, scaler, and adapter artifacts.
- `VALIDATION_RESULTS.md`: final package validation record.
- `SUBMISSION_CHECKLIST.md`: upload checklist.

Python source and `requirements.txt` are intentionally omitted because the official
submission is executable-only and must not depend on package installation.
