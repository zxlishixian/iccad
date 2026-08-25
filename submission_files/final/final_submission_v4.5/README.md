# Regression Failure Bucketing — final_submission_v4.5

> Intermediate version (4 faithful training sets, no v4 expansion). Official
> mean ≈ 0.721. The latest submission is `final_submission_v5` (adds the v4
> expansion batch, official mean ≈ 0.870).

## Package layout

```
final_submission_v4.5/
├── regr_fail_bucketing       # PyInstaller onedir executable (primary)
├── _internal/                # bundled runtime (interpreter, libs, model weights)
│   └── models/               # encoder_seed{0..4}.npz + preprocess.pkl (ship with binary)
├── regr_fail_bucketing.py    # source fallback (self-contained baseline)
├── models/                   # source-of-truth weights (redundant, not needed at runtime)
└── README.md
```

## Usage

```bash
./regr_fail_bucketing --input <input.csv> --output <output.csv> --k <k>
```

Output is `Case,bucket`. Only the LLM **embedding** endpoint is called (via
`LLM_MODEL_CONFIG`); on failure the LLM block is zero-filled and the model runs
deterministically.

## Model

Siamese per-case encoder (5-seed Procrustes-aligned average) → k-means. 364-dim
features: LLM embedding + failure signature + test-name category + sim.log UVM
fatal-line char n-gram + divergence-window + trace residual. CPU-only, O(N).

Trained on 2324 synthetic cases (catalog + benchmark5/8_500 + k32). Official-dev
clean-transfer score: set1≈0.722, set2≈0.720, mean ≈ 0.721.

## Recovery (if the executable cannot start)

The evaluator falls back to `python regr_fail_bucketing.py --input ... --output
... --k ...`. That file is self-contained and standard-library-only (Drain +
TF-IDF/hash + k-means/agglomerative), so it runs without numpy/sklearn and always
emits a valid CSV — a lower-scoring safety net that avoids a zero.
