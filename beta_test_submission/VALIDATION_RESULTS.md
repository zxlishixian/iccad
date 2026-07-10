# Beta Package Validation

Validation date: 2026-07-11

## Requirements and safety target

The package was checked against `Alpha Test Submission Guideline_ABC.pdf`, `B_20260601.pdf`, and `B_QA_20260612.pdf`: required top-level executable, self-contained Linux x86_64 deployment, no Docker, output format, and Google Drive folder delivery. Although Beta doubles the standard time limits, all new routing decisions below use the original Final limits as the guardrail.

The official metric is pairwise balanced accuracy averaged over ten benchmarks. A benchmark that fails or times out scores zero, so bounded fallback is part of the model design.

## Model protocol

The canonical multi-view route uses only sim/regr evidence and organizer embeddings. It has five views, 64-dimensional train-fitted reducers, five GBDT seeds, dual/five-view probability blend 0.50, seed probability averaging, fixed supplied `k`, and average linkage. Every training benchmark is an independent episode; pairs are sampled only within a dataset. Runtime does not discover or read gold/golden/meta/trace and never calls completion.

Canonicalization removes only arbitrary case-index prefixes from feature/summary documents. This creates exact duplicate documents that are embedded once. Artifacts were retrained with this representation; it is not an inference-only text rewrite.

Historical corrected seven-dataset episode LODO, seeds 0–4, for the original multi-view model:

| Method | Mean BA | Worst BA | Mean TPR | Mean TNR |
|---|---:|---:|---:|---:|
| dual | 0.738913 | 0.527778 | 0.634376 | 0.843450 |
| five-view | 0.785807 | 0.675312 | 0.699148 | 0.872465 |
| dual/five-view seed mean + average linkage | 0.7977 | 0.5278 | 0.7502 | 0.8451 |

Canonical-artifact external screening, each target dataset fully excluded from its single-seed training fold:

| Held-out dataset | BA | TPR | TNR |
|---|---:|---:|---:|
| stage2, 240 cases | 0.913667 | 0.916667 | 0.910667 |
| VCS, 40 cases | 0.742771 | 0.641791 | 0.843750 |
| directed, 37 cases | 0.690282 | 0.579832 | 0.800731 |

These three screens support the runtime artifact but are not a replacement for a full canonical 7-dataset, multi-seed LODO study.

## End-to-end package validation

The top-level router was invoked directly with the local OpenAI-compatible Nomic embedding endpoint. All five views returned 768-dimensional embeddings; no embedding fallback was used.

| Test | Cases | Route | Clusters | BA | TPR | TNR | Wall | Max RSS |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| benchmark_set_1 | 7 | canonical five-view | 2 | 1.000000 | 1.000000 | 1.000000 | 13.00 s | 184 MB |
| benchmark_set_2 | 25 | canonical five-view | 4 | 0.950469 | 0.982906 | 0.918033 | 17.42 s | 196 MB |
| cold 160-case stress | 160 | canonical five-view | 16 | n/a | n/a | n/a | 70.13 s | 473 MB |
| forced 1-second multi-view timeout | 160 | deterministic agglomerative | 14 | n/a | n/a | n/a | 3.88 s | 170 MB |

The 160-case cold run is below the original Final 100-second limit. The medium route has a 90-second process guard; on failure it immediately uses deterministic clustering, avoiding a second embedding call. All validated outputs are exactly `Case,bucket` and have one data row per input case.

## Binary compatibility and integrity

- PyInstaller 6.20.0 onedir applications.
- Packaged ELF files scanned: 826.
- Maximum required GLIBC symbol: 2.28; symbols above 2.28: 0.
- Symlinks after materialization: 0.
- Package size: approximately 519 MiB.
- Top-level router and multiview backend modes: 755.
- No `gold.csv`, `golden.csv`, `meta.csv`, trace logs, Python source files, or API configuration files are packaged.
- Credential-pattern hits were manually traced to required scikit-learn HTML/CSS representation assets; no API key or credential is packaged.

## Residual risks

- The package was symbol-scanned on Ubuntu, not executed on the organizer's actual RHEL 8 host.
- Organizer endpoint latency can differ from the local CPU endpoint; 24/90-second guards and deterministic fallback protect Final limits.
- Public labels and known fake labels informed model selection. Public scores are final-artifact sanity scores, not hidden-set generalization estimates.

## Final hashes

- Router SHA-256: `3ba0024ce225a37f4cf04b761ccf1cb4067a5775241e0f61d4364772e555a134`.
- Canonical multiview SHA-256: `f69eeae39db888f37a54dcd51f376e21cb340d1c8a7b257f0105d5ecc750779f`.
- Calibrated Alpha fallback SHA-256: `8772045d823720981a07d78413c6edced55877e074422c45391353c1084a91cd`.
- Deterministic fast backend SHA-256: `bf0369c1446ee1f51af2f3f684b8d22df1a92a73087ffe4f02b087364ff97322`.
