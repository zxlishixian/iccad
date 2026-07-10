# Beta Package Validation

Validation date: 2026-07-11

## Official requirements checked

Sources: `Alpha Test Submission Guideline_ABC.pdf`, `B_20260601.pdf`, and `B_QA_20260612.pdf`. The package follows the required top-level executable interface, Linux x86_64 and GLIBC 2.28 compatibility, self-contained no-install deployment, no-Docker rule, and Google Drive folder delivery format. The official machine provides 32 logical CPU cores and approximately 128 GB RAM. Beta timeout limits are twice the normal benchmark limits.

## Model and protocol

The new route uses sim/regr only, five embedding views, five GBDT seeds, dual/five-view blend `0.50`, seed probability averaging, fixed reference `k`, and average linkage. Training and LODO preprocessing build each benchmark independently. Runtime does not read gold/golden/meta/trace and does not call completion.

Corrected seven-dataset episode LODO, seeds 0-4:

| Method | Mean BA | Worst BA | Mean TPR | Mean TNR |
|---|---:|---:|---:|---:|
| dual | 0.738913 | 0.527778 | 0.634376 | 0.843450 |
| five-view | 0.785807 | 0.675312 | 0.699148 | 0.872465 |
| dual/five-view seed mean + average link | 0.7977 | 0.5278 | 0.7502 | 0.8451 |

The final artifacts are trained on the selected seven datasets after model selection. Public-set scores below are final-artifact sanity results, not held-out estimates.

## Binary compatibility

- PyInstaller 6.20.0 onedir applications.
- Packaged ELF files scanned: 826.
- Maximum required GLIBC symbol: 2.28; symbols above 2.28: 0.
- Symlinks after materialization: 0.
- Total files: 1825.
- Package size: approximately 520 MiB.
- Router and backend executable modes: 755.
- Router SHA-256: `902515919a70120292a6dd14643e6f2a69ea283f86a5377d020e989657071e2e`.
- Multi-view binary SHA-256: `0578a7be1610dfd2d10151f7bf1b721151bb844ad6a796c83d1828e36a3c3086`.
- Alpha fallback SHA-256: `8772045d823720981a07d78413c6edced55877e074422c45391353c1084a91cd`.
- Fast binary SHA-256: `bf0369c1446ee1f51af2f3f684b8d22df1a92a73087ffe4f02b087364ff97322`.

## End-to-end validation

The top-level router was invoked directly without an activated Python environment. The local OpenAI-compatible endpoint returned 768-dimensional features, summary, event, object, and context embeddings with no fallback.

| Dataset / test | Cases | Route | Clusters | BA | TPR | TNR | Wall | Max RSS |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| benchmark_set_1 | 7 | multi-view | 2 | 1.000000 | 1.000000 | 1.000000 | 11.78 s | 163 MB |
| benchmark_set_2 | 25 | multi-view | 4 | 0.950469 | 0.982906 | 0.918033 | 17.88 s | 172 MB |
| stage3 | 640 | deterministic agglomerative | 28 | 0.778823 | 0.847862 | 0.709783 | 5.97 s | 184 MB |
| repeated stage3 | 3000 | deterministic k-means | 64 | n/a | n/a | n/a | 19.48 s | 272 MB |

All outputs have exactly `Case,bucket`, the expected row count, and the expected route-specific cluster count where fixed k is used. The 3000-case input repeats only 640 unique case identifiers and is runtime-only.

## Failure and fallback validation

- Missing `LLM_MODEL_CONFIG`: multi-view is skipped; the bundled Alpha path completed set1 in 2.14 s and ultimately emitted a valid deterministic result.
- Forced one-second multi-view wall timeout: router printed a fallback warning, the Alpha model completed, and a valid output was produced in 3.82 s.
- Multi-view validates all five embedding matrices as 768-dimensional and exits nonzero on fallback vectors, allowing the router to recover.

## Residual risks

- The package was assembled on Ubuntu and symbol-scanned, but not executed on an actual RHEL 8 host.
- Official endpoint latency may differ from the local CPU-compatible endpoint; the wall-time guard protects small-set execution.
- Public labels were used in final training, so public sanity scores must not be interpreted as hidden-set generalization estimates.
