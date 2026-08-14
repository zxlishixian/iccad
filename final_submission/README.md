# Regression Failure Bucketing — final submission (siamese)

## Required interface

```bash
./regr_fail_bucketing --input <input.csv> --output <output.csv> --k <k>
```

`regr_fail_bucketing` is a PyInstaller onedir executable (self-contained, no
PyTorch / pip / GPU).  Output is exactly `Case,bucket` with one row per case.

## Model

O(N) siamese per-case encoder (5-seed embedding average) → k-means clustering.

Per-case features:
- LLM embedding (nomic-embed-text-v1.5, 768 → SVD 64)
- failure signature (functional-unit family + divergence type)
- failing-test-name semantic category flags (csr/interrupt/debug/mmu/...)
- hierarchical trace residual (96-dim)

The encoder is a NumPy reimplementation of the PyTorch model (max error ~8e-5);
no PyTorch is bundled.

## LLM configuration

Reads `LLM_MODEL_CONFIG` (YAML content). Only the `embedding` endpoint is
called. On missing config / API failure / incompatible dims, falls back to the
deterministic no-trace baseline and still writes a valid CSV.

## GLIBC compatibility

Built on a glibc 2.39 host.  The initially-bundled `libgcc_s.so.1` (GLIBC_2.35)
and `libstdc++.so.6` (GLIBC_2.38) were replaced with the conda `libgcc-15.2.0`
(GLIBC_2.14) and `libstdcxx-15.2.0` (GLIBC_2.17) packages.  The highest GLIBC
symbol across all bundled ELF files is now **GLIBC_2.28**, which meets the
official ≤ 2.28 ceiling.

## Local validation (train-on-dev, upper bound)

| Dataset | k | BA | wall |
|---|---|---:|---:|
| benchmark_set_1 | 2 | 0.83 | 2.9 s |
| benchmark_set_2 | 4 | 0.90 | 3.9 s |
| official_vcs | 3 | 0.80 | 3.3 s |
| directed_cross_v4 | 10 | 0.78 | 4.6 s |
| stable | 4 | 0.78 | 2.9 s |
| benchmark6 | 64 | 0.56 | 70.6 s |

Note: these are train-on-dev scores (the official dev sets were in the training
data) and are an upper bound; the hidden-set generalization is expected to be
around the alpha submission's ~0.72 level.  All runtimes are within the official
100 s / 300 s limits for N=3000.
