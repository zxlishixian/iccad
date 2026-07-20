# Beta Package Validation

Validation date: 2026-07-13

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

The 160-case cold run is below the original Final 100-second limit. Runtime routing combines actual case-count scale, soft-`k` scale, and a metadata-only context estimate. The larger case/`k` scale wins. Max-10/30 scales reserve 7 seconds for baseline and 18 seconds for five-view. Non-100M-like max-100 and max-300/1000/3000 scales use 12+72 and 12+70 seconds respectively; 100M-like hidden inputs use 15+60 seconds. The official Q&A exposes no authoritative 10M/100M tier marker, so the package never assumes a 300-second allowance.

Case count and soft `k` are independently mapped to the official 10/30/100/300/1000/3000-case and 2/4/8/16/32/64-bucket scales; the larger scale wins. Tests confirmed: 7 cases with `k=2` resolves to max-10/30 seconds, the same 7 cases with `k=8` resolves to max-100/100 seconds, and 240 cases with undersized `k=8` resolves to max-300/100 seconds.

Context-line estimation uses filesystem sizes only and does not open trace logs. On the public sets it estimated 62,668 and 676,277 lines respectively; both selected `one_m_like` and the 30-second class. Forced tests confirmed that a public-shaped context anomaly remains capped at 30 seconds, while `hundred_m_like` hidden input selects `extended_context_conservative` and remains capped at 100 seconds.

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
- Packaged ELF files scanned: 827.
- Maximum required GLIBC symbol: 2.28; symbols above 2.28: 0.
- Symlinks after materialization: 0.
- Package size: approximately 943 MiB apparent file bytes (about 536 MiB allocated locally).
- Top-level router and multiview backend modes: 755.
- No `gold.csv`, `golden.csv`, `meta.csv`, trace logs, Python source files, or API configuration files are packaged.
- Credential-pattern hits were manually traced to required scikit-learn HTML/CSS representation assets; no API key or credential is packaged.

## Residual risks

- The package was symbol-scanned on Ubuntu, not executed on the organizer's actual RHEL 8 host.
- Organizer endpoint latency can differ from the local CPU endpoint; scale-aware 18/72/70/60-second expert guards, an already-published deterministic result, exact result validation, and a singleton writer protect Final limits.
- The frozen five-view artifacts expect 768-dimensional embeddings. An organizer endpoint that returns a different dimension is treated as incompatible; the deterministic result remains valid, but that endpoint run will not receive the five-view improvement.
- The interface provides no authoritative 10M/100M tier marker. The metadata estimate may be noisy, so it is used only to shorten the primary attempt; all hidden routes remain capped at 100 seconds.
- Public labels and known fake labels informed model selection. Public scores are final-artifact sanity scores, not hidden-set generalization estimates.

## Final hashes

- Entry SHA-256: `f6fadf592fb3ff2468d846c5b6d5c37a1bd10da62da0420ffdadb40c9992f64d`.
- Router policy SHA-256: `27c009f33d02b6681eb6cc126051d9cb74f683dcfff1c911bd020d803e592dae`.
- Sparse multiview SHA-256: `a3ecec3e411e29d21b377d45da873a84d2fcf5c474973e3a28cb7d7f05854ff9`.
- Canonical multiview SHA-256: `f69eeae39db888f37a54dcd51f376e21cb340d1c8a7b257f0105d5ecc750779f`.
- Calibrated Alpha fallback SHA-256: `8772045d823720981a07d78413c6edced55877e074422c45391353c1084a91cd`.
- Deterministic fast backend SHA-256: `bf0369c1446ee1f51af2f3f684b8d22df1a92a73087ffe4f02b087364ff97322`.

## Previous official-scale sparse routing validation (superseded, 2026-07-13)

The superseded expert thresholds mapped directly to the official case/`k` scales. The 100-case scale uses full five-view inference, the 300-case scale uses deterministic-first sparse five-view refinement, and the 1000/3000-case scales retain deterministic k-means. All timings target the original Final limits.

| Runtime-only profile | Cases | k | Selected route | Wall | Rows | Result |
|---|---:|---:|---|---:|---:|---|
| B4-shaped | 300 | 16 | sparse five-view | 70.31 s | 301 | expert completed |
| B5-shaped | 1000 | 32 | deterministic k-means | 6.88 s | 1001 | valid baseline |
| B6-shaped | 3000 | 64 | deterministic k-means | 8.22 s | 3001 | valid baseline |

On the 240-case labeled stage2 runtime check, the same sparse route changed BA from `0.7657` to `0.7772`. This score is a package sanity result, not an independent generalization estimate, because the final package artifact used that training family. Independent five-domain LODO simulation for the new centroid selector improved macro BA from `0.6601` to `0.6739`, but a 640-case sanity check still traded too much TPR for TNR (`0.7788` baseline BA versus `0.7703` conservative centroid BA). Therefore centroid refinement above 300 cases remains experimental and disabled by default.

The router writes singletons immediately, atomically publishes a validated deterministic candidate, and only then runs the expert. Watchdogs terminate the complete expert process group. A sparse timeout or malformed output cannot erase the deterministic CSV.


## Experimental dual-view scale-extension rejection (2026-07-13)

This experiment did not change the packaged route. A dual-only sparse expert
used the same deterministic selector and refinement rule as the five-view
candidate. Independent five-domain LODO macro BA was `0.6802`, compared with
`0.6798` for the matched five-view policy and `0.6601` for its deterministic
proxy. However, the exact cap-20 policy was neutral on held-out stage3
(`0.7181 -> 0.7179`), while a package-artifact 640-case run fell to `0.7577`
and required `399.2 s`. A 1000-case standalone cold stress exceeded `110 s`.

Historical decision (now superseded by the all-scale feedback build): keep 1000/3000-case expert routing disabled. Those scales previously received the immediately published emergency output followed by the bounded deterministic k-means attempt. The dual source path remains an experiment for a
future single-parse implementation; it is not part of the hashes above.

## All-scale canonical five-view feedback build (2026-07-13)

The default Beta v2 route now attempts the score-leading canonical five-view
backend for every official case/`k` scale: 10/2, 30/4, 100/8, 300/16,
1000/32, and 3000/64. This is an intentional score-first experiment for
organizer-machine timing feedback. The sparse and calibrated-dual binaries are
retained for rollback but are not selected by the default route.

The anytime contract is unchanged: singleton output is written first, a
validated deterministic candidate is atomically published second, and the
five-view result replaces it only after exact header/row-count validation.
The existing Final-time watchdog remains active, so a slow, failed, malformed,
or dimension-incompatible five-view attempt preserves the deterministic CSV.

Nine router tests pass, including mock 100/300/1000/3000-case five-view
selection and large-expert timeout recovery. A real package smoke on public
set1 confirmed five 768-dimensional views with no fallback, selected
`multiview_all_scales`, wrote a valid 8-line `Case,bucket` CSV, and completed in
3.93 seconds. Large-scale five-view completion time remains deliberately
unverified locally; organizer feedback is the purpose of this build.
