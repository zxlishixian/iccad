# ICCAD 2026 Problem B Alpha Submission

## Interface

```bash
./regr_fail_bucketing --input <input.csv> --output <output.csv> --k <k>
```

The output is a CSV with exactly two columns:

```text
Case,bucket
```

## Submitted model

Primary prediction path:

1. Read `sim.log(.gz)` and `regr.log` referenced by `input.csv`.
2. Build deterministic SVD features and two LLM embedding views:
   `features` and `summary`.
3. Run the 10-seed no-trace calibrated dual-input pairwise blend:
   - rich residual focal MLP;
   - logistic/GBDT/shallow-MLP soft-voting ensemble;
   - rich/ensemble weights `0.88/0.12`;
   - rich temperature `1.15`;
   - ensemble temperature `1.00`.
4. Apply the lightweight official-style root-cause logistic adapter with
   blend alpha `0.50`.
5. Cluster the pairwise probability matrix using average-linkage
   agglomerative clustering and the supplied `k`.

The adapter artifact was trained before packaging. Prediction does not read
`gold.csv`, `golden.csv`, `meta.csv`, or `trace.log`.

## LLM configuration

The program reads YAML content from `LLM_MODEL_CONFIG`, as specified by the
contest:

```python
yaml.safe_load(os.getenv("LLM_MODEL_CONFIG"))
```

Only the `embedding` endpoint is used. Completion is never called.

## Failure handling

If the embedding endpoint, a model artifact, or an optional dependency fails,
the entrypoint falls back to the deterministic no-trace pipeline and still
writes a valid output CSV.

## Public validation provenance

Leave-one-official-benchmark-out validation of the official-style logistic
route:

| Train | Test | BA | TPR | TNR |
|---|---|---:|---:|---:|
| benchmark_set_2 | benchmark_set_1 | 0.7222 | 0.7778 | 0.6667 |
| benchmark_set_1 | benchmark_set_2 | 0.9072 | 0.9402 | 0.8743 |

Adding the calibrated no-trace blend on benchmark_set_2 produced BA `0.9213`.
These public results are validation references, not guarantees for hidden
benchmarks.

## Files

- `regr_fail_bucketing`: required launcher.
- `regr_fail_bucketing.bin`: standalone executable when present.
- `submission_main.py`: source fallback and primary orchestration.
- `regr_fail_bucketing.py`: deterministic baseline and embedding client.
- `pairwise_features.py`, `pairwise_llm_features.py`: pairwise model features.
- `official_style_features.py`: root-cause adapter features.
- `trace_anchor.py`, `trace_features.py`: parsing helpers; trace is not read by
  the submitted prediction path.
- `models/`: frozen inference artifacts.
- `requirements.txt`: source-fallback dependencies.
