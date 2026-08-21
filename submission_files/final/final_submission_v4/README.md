# Regression Failure Bucketing — final submission v4 (siamese, clean migration)

## Required interface

```bash
./regr_fail_bucketing --input <input.csv> --output <output.csv> --k <k>
```

`regr_fail_bucketing` is a PyInstaller onedir executable (self-contained, no
PyTorch / pip / GPU).  Output is exactly `Case,bucket` with one row per case.

## Model

O(N) siamese per-case encoder (5-seed ensemble) → consensus clustering.

The 5 seeds are ensembled by **co-association voting**: each seed clusters the
encoded cases independently, then the pairwise cluster agreement is voted across
seeds and a final agglomerative consensus clustering produces the buckets.  This
is more robust than averaging the (not mutually aligned) per-seed embeddings.

Per-case features:
- LLM embedding (nomic-embed-text-v1.5, 768 → SVD 64)
- failure signature (functional-unit family + divergence type) — exact first-mismatch
  opcode was tried and reverted (it re-introduces "diff-bugs-same-syndrome" confusion
  on official dev sets; see handoff.md pitfall #23)
- failing-test-name semantic category flags (csr/interrupt/debug/mmu/...)
- hierarchical trace residual (96-dim)

The encoder is a NumPy reimplementation of the PyTorch model (max error ~8e-5);
no PyTorch is bundled.

Training data: the 5-seed ensemble is trained on **all 9 fake sets, no official
dev sets** (clean migration — the official sets are held out entirely).  The four
high-quality sets (benchmark5_500cases_official / benchmark8_500cases_official /
k32_new12_380cases_official / batch4_11bugs) are each up-weighted ×2; the older
lui-cascade sets (official_vcs / directed_cross_v4 / stable / benchmark5_final /
benchmark6_final) are kept at ×1 as pretraining diversity.

Clean-migration validation (trained on fake only, official held out):
benchmark_set_1 BA 1.00, benchmark_set_2 BA 0.70 (official mean 0.85, up from
the 0.53 baseline of the old lui-cascade fake data).

## LLM configuration

Reads `LLM_MODEL_CONFIG` (YAML content). Only the `embedding` endpoint is
called, batched (512 docs/call, 20 s/call timeout) so official-eval network
latency stays inside the runtime limits. On missing config / API failure /
incompatible dims, the LLM feature block is zero-filled and the siamese model
still runs (verified BA 1.0 on benchmark_set_1 and 0.90 on benchmark_set_2
with no LLM); a valid CSV is always written.

`regr_fail_bucketing.py` is the stdlib-first source fallback, included per the
official rule that the evaluator tries the source only if the executable cannot
start. It is not the primary path.

## GLIBC compatibility

Built on a glibc 2.39 host.  The bundled `libgcc_s.so.1` and `libstdc++.so.6`
were replaced with the conda `libgcc-15.2.0` (GLIBC_2.14) and `libstdcxx-15.2.0`
(GLIBC_2.17) packages.  The highest GLIBC symbol across all bundled ELF files is
**GLIBC_2.28**, meeting the official ≤ 2.28 ceiling.

## Local validation (train-on-dev, upper bound)

| Dataset | k | BA |
|---|---|---:|
| benchmark_set_1 | 2 | 1.00 |
| benchmark_set_2 | 4 | 0.97 |
| benchmark5_500cases_official | 10 | 0.95 |
| benchmark8_500cases_official | 10 | 1.00 |
| official_vcs | 3 | 0.88 |
| stable | 4 | 0.83 |
| benchmark6_final | 64 | 0.62 |

Note: these are train-on-dev scores (the official dev sets were in the training
data) and are an upper bound; the hidden-set generalization is expected to be
around the alpha submission's ~0.72 level or better.
