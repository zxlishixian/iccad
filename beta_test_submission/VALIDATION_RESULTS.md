# Packaged Model Validation

Validation date: 2026-06-15

## Official requirements checked

Sources: `Alpha Test Submission Guideline_ABC.pdf`, `B_20260601.pdf`, and
`B_QA_20260612.pdf`. The package follows the required executable name/interface,
RHEL/Alma/CentOS 8-compatible GLIBC 2.28 ceiling, Linux x86_64 target, self-contained
no-install deployment, no-Docker rule, and Google Drive folder delivery format.

The official machine has no GPU, 32 logical CPU cores, approximately 128 GB RAM,
and no ordinary internet access. LLM calls use the organizer-provided HTTP endpoint.
Alpha timeouts are 3x the normal limits: public sets are 90 seconds each; hidden sets
are 300 or 900 seconds according to the benchmark table.

## Binary validation

- PyInstaller 6.20.0 onedir executable.
- Runtime model implementation: NumPy MLP + sklearn logistic/GBDT.
- Original PyTorch versus NumPy max logit error over all 20 MLP artifacts: `9.54e-7`.
- Packaged ELF files scanned: 287.
- Maximum required GLIBC symbol: `2.28`.
- Symlinks after materialization: 0.
- Executable mode: 755.
- Executable SHA-256:
  `8772045d823720981a07d78413c6edced55877e074422c45391353c1084a91cd`.

## End-to-end public validation

The binary was invoked directly, without `PYTHON_BIN` or an activated environment.
The local endpoint returned 768-dimensional `features` and `summary` embeddings with
no fallback warning.

| Dataset | Cases | k | Clusters | BA | TPR | TNR | Wall | Max RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| benchmark_set_1 | 7 | 2 | 2 | 0.722222 | 0.777778 | 0.666667 | 2.55 s | 134 MB |
| benchmark_set_2 | 25 | 4 | 4 | 0.921255 | 0.957265 | 0.885246 | 6.33 s | 153 MB |

The resulting CSV files are byte-identical to the previous validated PyTorch package.

## Fallback validation

With `LLM_MODEL_CONFIG` removed, primary inference failed closed and the bundled
deterministic no-trace fallback completed benchmark_set_1 in 2.44 seconds. It wrote
`Case,bucket` plus seven case rows and exited with status 0.

## Residual risks

- The package was assembled on Ubuntu using Conda binaries whose ELF symbol ceiling
  was explicitly verified at GLIBC 2.28; it was not executed on an actual RHEL 8 host.
- Hidden 1000/3000-case benchmarks were not available for end-to-end runtime testing.
  Pair feature construction is O(N^2), so large hidden-set runtime remains the main
  operational risk.
- The provided submission deadline is June 12, 2026 at 17:00 GMT+8; acceptance after
  that timestamp requires organizer confirmation.

## Beta timeout-aware routing validation

The beta candidate keeps the full Alpha model for small/medium public-style sets
and adds a deterministic fast backend for larger sets. This is intended to avoid
Alpha's slow-pass behavior on large hidden benchmarks while preserving public
set scores.

| Dataset / stress test | Cases | Route | Wall | Max RSS | BA | TPR | TNR |
|---|---:|---|---:|---:|---:|---:|---:|
| benchmark_set_1 | 7 | full model | 2.69 s | 131 MB | 0.722222 | 0.777778 | 0.666667 |
| benchmark_set_2 | 25 | full model | 6.74 s | 149 MB | 0.921255 | 0.957265 | 0.885246 |
| old fake stage3 | 640 | fast deterministic agglomerative | 10.87 s | 182 MB | 0.778823 | 0.847862 | 0.709783 |
| repeated stage3 stress | 3000 | fast deterministic k-means | 28.07 s | 307 MB | n/a | n/a | n/a |

The 3000-case stress input repeats existing stage3 rows with absolute paths. It
is a runtime-only check, not a meaningful scoring benchmark.
