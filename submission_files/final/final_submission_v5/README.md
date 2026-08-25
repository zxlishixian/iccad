# Regression Failure Bucketing — final_submission_v5

Buckets RTL regression failure cases by root cause using a siamese encoder +
k-means. This is a **clean-transfer** model (trained only on synthetic data, no
official dev sets).

## Package layout

```
final_submission_v5/
├── regr_fail_bucketing       # PyInstaller onedir executable (primary entry point)
├── _internal/                # bundled runtime (Python interpreter, libs, model weights)
│   └── models/               # encoder_seed{0..4}.npz + preprocess.pkl (must ship with the binary)
├── regr_fail_bucketing.py    # source fallback (self-contained baseline, see "Recovery")
├── models/                   # source-of-truth model weights (redundant copy, not needed at runtime)
└── README.md
```

The executable is self-contained: it bundles its own Python runtime and
libraries, so it runs without pip / network / GPU / numpy-sklearn on the host.
The model weights live in `_internal/models/` and are loaded relative to the
executable — **always submit the `regr_fail_bucketing` file together with the
`_internal/` directory**.

## Usage

```bash
./regr_fail_bucketing --input <input.csv> --output <output.csv> --k <k>
```

- `input.csv` — three columns `Case,Regr Log,Sim Log,Trace Log` (log paths).
- `--k` — number of buckets (= number of injected bugs).
- Output is `Case,bucket` (one row per case; header is case-insensitive per the
  official Q&A).

Only the LLM **embedding** endpoint is called, and only if `LLM_MODEL_CONFIG`
(YAML content) is present. On a missing config / API error / wrong dims, the
LLM feature block is zero-filled and the model still runs deterministically —
there is no hard dependency on the LLM.

## Model

- Per-case features (364 dims): LLM embedding (768→SVD 64) + failure signature
  (functional-unit family + divergence type) + test-name category flags +
  sim.log first UVM_FATAL/ERROR line char n-gram (128) + divergence-window
  distribution (48) + hierarchical trace residual (96).
- 5-seed ensemble: per-seed embeddings are Procrustes-aligned then averaged,
  then k-means into exactly `k` clusters.
- CPU-only; O(N) inference; well inside the 100s/300s limits even at N=3000.

Trained on 3268 synthetic cases (v4 expansion "1 bug = multiple tests" +
catalog + benchmark5/8_500 + k32). Official-dev clean-transfer score (packaged
binary, with LLM): **set1=1.000, set2=0.741, mean ≈ 0.870**.

## Recovery (if the executable cannot start)

The official evaluator falls back to source **only if the executable cannot
start**. In that case it runs:

```bash
python regr_fail_bucketing.py --input <input.csv> --output <output.csv> --k <k>
```

`regr_fail_bucketing.py` is a **self-contained, standard-library-only** baseline
(Drain log templating + TF-IDF/hash features + k-means/agglomerative). It has
zero third-party dependencies, so it runs on the evaluator even when numpy /
sklearn / pandas are absent. It implements the same interface and always emits a
valid `Case,bucket` CSV. It scores lower than the binary (it is a baseline), but
guarantees a non-zero score instead of a crash.

To maximise the score, submit the executable and `_internal/` (primary); the
source file is only a safety net.
