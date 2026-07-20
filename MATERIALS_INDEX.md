# ICCAD Project Materials Index

This repository contains the code, datasets, experiment records, submission
packages, and competition documents for the ICCAD 2026 Problem B regression
failure bucketing project.

## Start Here

- `README.md`: project history, methods, validation results, and current status.
- `ICCAD阶段性成果报告.md`: Chinese milestone report.
- `regr_fail_bucketing.py`: stable deterministic reference implementation.
- `regr_fail_bucketing`: command-line entry used during development.

## Competition Documents

- `B_20260601.pdf`: Problem B specification, scoring, interface, and limits.
- `B_QA_20260612.pdf`: official environment and packaging clarifications.
- `Alpha Test Submission Guideline_ABC.pdf`: submission workflow.
- `cadb1001.docx`: Alpha evaluation feedback material.
- `benchmark_manual_label/`: historical manual analysis of public benchmarks.

## Datasets

- `test_case/`: official public benchmark inputs and released golden labels.
- `old_fake_dataset/`: the three original synthetic research datasets.
- `official_format_fake_dataset/`: newer official-format synthetic datasets.
- `runtime_only_benchmarks_unique_v2/`: runtime-only stress data; it is not used
  for score claims or supervised training.

Every benchmark is treated as an independent episode. Training and evaluation
scripts must not construct labeled pairs across benchmark boundaries.

## Main Experimental Code

- `pairwise_llm_features.py`, `train_pairwise_llm.py`: pairwise feature and
  supervised model pipeline.
- `multigranular_features.py`, `multiview_case_features.py`: multi-view log
  evidence extraction.
- `run_*experiments.py`: reproducible training, LODO, calibration, trace, and
  clustering experiments.
- `trace_*.py`: experimental trace parsing and representation routes.
- `tests/`: routing, feature, clustering, and fallback tests.
- `tools/`: local endpoint compatibility and packaging helpers.

## Submission Packages

- `alpha_test_submission/`: frozen Alpha submission package.
- `beta_test_submission/`: earlier Beta candidate.
- `beta_test_submission_v2/`: Final-time-aware, baseline-first Beta v2 package.
- `beta_test_submission_v3/`: latest renamed experimental submission package and
  LODO artifacts.

Submission packages contain self-contained PyInstaller backends. Their bundled
runtime files are intentionally versioned so that an exact submitted package
can be reconstructed. See each package's `README_SUBMISSION.md`,
`SUBMISSION_CHECKLIST.md`, and `VALIDATION_RESULTS.md` before use.

## Security And Local State

API credentials, `.env` files, Codex authentication files, virtual
environments, and local model caches are not repository materials and must not
be committed. Official LLM credentials are supplied at evaluation time through
the YAML content in `LLM_MODEL_CONFIG`.

Large transient experiment outputs should remain under `/tmp`; only compact
summaries, source code, frozen artifacts, and submission packages belong in the
repository.
