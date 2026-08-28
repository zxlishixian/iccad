# Regression Failure Bucketing — final_submission_v8 (siamese, truncated-trace)

Buckets RTL regression failure cases by root cause using a siamese encoder +
k-means. Trained on the deduplicated synthetic release plus the two public
official dev sets. This version removes the `case_index` row number from the LLM
document (Section 3.7 compliance — no metadata dependency), and truncates trace
parsing to the last 5000 instructions to stay inside the runtime limits on large
benchmarks.

## Package layout

```
final_submission_v8/
├── regr_fail_bucketing       # PyInstaller onedir executable (primary entry point)
├── _internal/                # bundled runtime (Python interpreter, libs, model weights)
│   └── models/               # encoder_seed{0..4}.npz + preprocess_seed{0..4}.pkl
├── regr_fail_bucketing.py    # source fallback (self-contained baseline, see "Recovery")
└── README.md
```

The executable is self-contained: it bundles its own Python runtime and
libraries, so it runs without pip / network / GPU / numpy-sklearn on the host.
Always submit `regr_fail_bucketing` together with the `_internal/` directory.

## Usage

```bash
./regr_fail_bucketing --input <input.csv> --output <output.csv> --k <k>
```

- `input.csv` — a `Case` column plus three log-path columns `Regr Log,Sim Log,Trace Log`.
- `--k` — number of buckets (= number of injected bugs).
- Output is `Case,bucket` (one row per case; header is case-insensitive per the
  official Q&A).

Only the LLM **embedding** endpoint is called, and only if `LLM_MODEL_CONFIG`
(YAML content) is present. On a missing config / API error / wrong dims, the
LLM feature block is zero-filled and the model still runs deterministically.

## Model

- Per-case features (364 dims): LLM embedding (768→SVD 64) + failure signature
  (functional-unit family + divergence type) + test-name category flags +
  sim.log first UVM_FATAL/ERROR line char n-gram (128) + divergence-window
  distribution (48) + hierarchical trace residual (96).
- **Trace parsing is truncated to the last 5000 instructions** (a fatal-type
  bug's trace can grow to hundreds of thousands of lines by looping to timeout;
  the tail is where the discriminative signal lives, and the head is noise).
- 5-seed ensemble: each seed's LLM/trace reducers are paired with its own
  encoder (the reducers are fit with `random_state=seed`, so they are not
  shared across seeds); per-seed embeddings are Procrustes-aligned then averaged,
  then k-means into exactly `k` clusters. The NumPy forward pass uses exact GELU
  (erf), matching the PyTorch training semantics.
- CPU-only; O(N) inference; well inside the 100s/300s limits even at N=3000.

Training data (weighted): deduplicated_release_20260828_v3 (1376 cases / 118
bugs) + large_expansion ×2 (944×2) + catalog + benchmark5/8_500 + k32_new12,
plus the two public official dev sets (benchmark_set_1 + benchmark_set_2).

Official-dev score (packaged binary, with LLM): **set1=1.000, set2=0.979, mean
≈ 0.990**. Without LLM (fallback): set2 ≈ 0.733.

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
