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

The 160-case cold run is below the original Final 100-second limit. Runtime routing now follows both observable size axes. Public-size inputs (`n <= 30`, `k <= 4`) reserve 18 seconds plus a 7-second deterministic retry. Hidden inputs use the shortest applicable 100-second Final cap: 85+10 seconds normally, or 75+15 seconds when a metadata-only estimate exceeds 20M context lines. The official Q&A exposes no 10M/100M tier marker, so the package does not assume a 300-second allowance.

Context-line estimation uses filesystem sizes only and does not open trace logs. On the public sets it estimated 62,668 and 676,277 lines respectively; both correctly selected the 30-second class. A forced extended-context test selected `extended_context_conservative` and still reported `final_limit=100s`.

## Failure-injection fallback validation

The following tests invoked the top-level executable, never read labels at runtime, and verified the exact `Case,bucket` header and input-row count.

| Injection | Cases | Expected recovery | Observed result | Wall |
|---|---:|---|---|---:|
| no `LLM_MODEL_CONFIG` | 7 | deterministic agglomerative | valid 8-line CSV, exit 0 | 6.80 s |
| multi-view watchdog = 1 s | 7 | deterministic agglomerative | valid 8-line CSV, exit 0 | 2.72 s |
| multi-view watchdog = 1 s | 37 | fast then singleton if fast exceeds 10 s | valid 38-line singleton CSV, exit 0 | 10.91 s |
| calibrated-dual watchdog = 1 s | 240 | deterministic agglomerative | valid 241-line CSV, exit 0 | 7.58 s |
| extended-context + multi-view watchdog = 1 s | 37 | deterministic agglomerative | valid 38-line CSV, exit 0 | 6.97 s |
| fast binary deliberately absent | 7 | singleton emergency writer | valid 8-line singleton CSV, exit 0 | < 1 s |
| force k-means route | 640 | deterministic k-means | valid 641-line CSV, exit 0 | 5.32 s |

The singleton writer uses only the first `Case` column of the supplied input and makes no quality claim: it normally has TNR = 1 and TPR = 0, hence is generally preferable to the official zero awarded for a failed or timed-out benchmark. It cannot recover from an unreadable input, unwritable output directory, or an external evaluator that kills the entire router before its own guard executes.

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
- Organizer endpoint latency can differ from the local CPU endpoint; 18/85/75-second primary guards, bounded deterministic retries, result validation, and a singleton writer protect Final limits.
- The interface provides no authoritative 10M/100M tier marker. The metadata estimate may be noisy, so it is used only to shorten the primary attempt; all hidden routes remain capped at 100 seconds.
- Public labels and known fake labels informed model selection. Public scores are final-artifact sanity scores, not hidden-set generalization estimates.

## Final hashes

- Router SHA-256: `42e387328dcb9fc90f2f0571b46c9b601eb4f03b24def4c38bb9a1ce8f0d1cb4`.
- Canonical multiview SHA-256: `f69eeae39db888f37a54dcd51f376e21cb340d1c8a7b257f0105d5ecc750779f`.
- Calibrated Alpha fallback SHA-256: `8772045d823720981a07d78413c6edced55877e074422c45391353c1084a91cd`.
- Deterministic fast backend SHA-256: `bf0369c1446ee1f51af2f3f684b8d22df1a92a73087ffe4f02b087364ff97322`.
