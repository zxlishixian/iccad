# Regression Failure Bucketing — final submission

Entry point:

```bash
regr_fail_bucketing --input IN.csv --output OUT.csv --k K
```

## Behavior

The `regr_fail_bucketing` wrapper always writes a valid `Case,bucket` CSV:

1. It writes a **singleton fallback** (one bucket per case) to `OUT.csv` first.
2. It runs `final_inference.py`, the 9-fold TriLog two-tower ensemble:
   - reconstructs the held-out pair-feature matrix (LLM feature/summary
     reducers, event/object/context view reducers, hierarchical trace-view
     reducers);
   - averages the pairwise probability matrices across seeds and folds;
   - clusters with weighted correlation clustering (degenerate fallback to
     agglomerative).
3. If the ensemble raises for any reason, `final_inference.py` falls back to
   the deterministic `regr_fail_bucketing.py` baseline (Drain + TF-IDF/SVD64 +
   agglomerative), which needs no model, trace, or LLM.
4. A watchdog kills the Python process at the runtime budget; if it has not
   finished, the singleton fallback is preserved.

## Model directory

`manifest.json` and `models/` (one `model_<fold>_seed<seed>.pt` plus one
`preprocess_<fold>_seed<seed>.pkl` per fold×seed) are produced by
`run_final_submission_train.py` and copied here by `package_final_submission.py`.

## LLM embeddings

LLM embeddings are read from the `LLM_MODEL_CONFIG` environment variable
(optional). If it is absent or the endpoint fails, features degrade to the
deterministic SVD-only representation, and on total failure the deterministic
baseline still produces a valid output.
