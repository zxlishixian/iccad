# Multi-View Generalization and Leakage Audit (2026-07-13)

## Verdict

The runtime predictor is compliant with the intended no-trace path, but the
initial seven-dataset LODO estimate was not fully independent. The three old
fake datasets are nested copies and must be treated as one data-source family.

The previously reported seven-fold macro BA 0.8516 is therefore an optimistic
diagnostic and must not be cited as strict seven-domain generalization.

## Runtime compliance

A system-call audit of beta_multiview_inference.py on official benchmark set1
opened only input.csv, referenced sim/regr logs, model files, libraries, and
embedding-cache files. It did not open gold.csv, golden.csv, meta.csv, or
trace.log.

All five views used the configured 768-dimensional embedding endpoint without
fallback. Cache entries contain only the embedding model name and numeric
embedding vector. Randomly permuting input rows changed 0/21 set1 pair
assignments and 0/300 set2 pair assignments, so prediction is not using case
order as a hidden feature.

The evaluation runner reads labels only to obtain the evaluation k and compute
BA/TPR/TNR. The formal predictor receives k on its CLI and does not read labels.

## Content-overlap finding

Exact model-visible sim+regr content overlap:

| datasets | exact shared cases | label agreement |
| --- | ---: | ---: |
| first_batch vs stage2 | 80 | 80/80 |
| first_batch vs stage3 | 80 | 80/80 |
| stage2 vs stage3 | 240 | 240/240 |

Thus first is contained in both larger datasets, and every stage2 case is
contained in stage3. Dataset-name LODO across these three leaks exact test cases.

The fake labels are also stored in contiguous runs. An order-only clustering
baseline obtains BA 0.8033/0.8042/0.8218 on first/stage2/stage3. The predictor is
order invariant, but these datasets remain weak evidence for model selection.

No exact raw-log duplicates were found between the official benchmarks and the
old fake family. Official set1/set2, VCS, and directed nevertheless have many
near-identical normalized failure templates; transfer among them is useful but
is an optimistic same-domain estimate.

## Layered held-out results

All rows use five-seed probability averaging, fixed k, real five-view
embeddings, and no target labels during training.

| training sources | held-out set1 BA | held-out set2 BA | official mean |
| --- | ---: | ---: | ---: |
| old family only (stage3) | 0.7222 | 0.8920 | 0.8071 |
| stage3 + VCS + directed; both official sets held out | 1.0000 | 0.8604 | 0.9302 |
| other six datasets; one official set held out | 1.0000 | 0.9201 | 0.9600 |

The first row is the cleanest cross-domain lower-bound estimate. The second
measures synthetic official-style transfer. The third measures public
official-to-official adaptation and is the most optimistic.

Set1 contains only seven cases, so BA 1.0 has high statistical uncertainty.
Repeated architecture, sampling, beta, and seed choices on the same public
benchmarks also create model-selection overfitting even when fold training is
technically held out.

## Guard fix

evaluation_leakage_guard.py now checks:

1. dataset name;
2. input.csv SHA-256;
3. per-case SHA-256 of model-visible sim+regr text.

train_multiview_submission.py records all three identities in new manifests.
run_leakage_safe_multiview_evaluation.py rejects overlap by default.
run_strict_multiview_lodo.py performs the content check before training.
The old fake seven-fold invocation now fails before model fitting and reports
the shared cases.

## Recommendation

Do not claim BA 0.9600 as expected hidden-test performance. A defensible current
range is:

- approximately 0.807 on strong cross-domain public holdout;
- up to 0.930 on official-style synthetic transfer;
- 0.960 only as an optimistic public-domain cross-validation estimate.

Final model selection should group first/stage2/stage3 as one source family,
deduplicate cases before sampling, keep both official public benchmarks out of
one final model-selection fold, and reserve a newly generated, provenance-clean
official-style dataset for one-time evaluation.
