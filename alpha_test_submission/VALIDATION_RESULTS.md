# Packaged Model Validation

Validation date: 2026-06-06

Environment:

- Python 3.12
- scikit-learn 1.8.0
- NumPy 2.4.4
- PyTorch 2.7.1
- embedding model `nomic-embed-text-v1.5`
- both `features` and `summary` embedding paths returned 768 dimensions
- no embedding fallback warning

Command:

```bash
./regr_fail_bucketing \
  --input test_case/problem/benchmark_set_1/input.csv \
  --output /tmp/set1.csv \
  --k 2
```

Public golden validation:

| Dataset | Cases | k | Predicted clusters | BA | TPR | TNR | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|
| benchmark_set_1 | 7 | 2 | 2 | 0.722222 | 0.777778 | 0.666667 | 5.11 s |
| benchmark_set_2 | 25 | 4 | 4 | 0.921255 | 0.957265 | 0.885246 | 9.43 s |

The wall times include Python startup, embedding requests, ten model seeds,
adapter inference, clustering, and CSV output. Both are below the 90-second
Alpha limit for the two public benchmarks.

Fallback validation:

- `LLM_MODEL_CONFIG` removed.
- Primary route failed closed because embedding vectors were unavailable.
- Deterministic no-trace fallback completed successfully.
- Output retained the required `Case,bucket` header and one row per case.

The public adapter was trained with released public labels. The packaged
prediction path contains only the frozen model artifact and never loads label
files.
