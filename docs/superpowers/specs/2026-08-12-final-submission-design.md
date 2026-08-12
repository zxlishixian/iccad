# Final Submission Design: Theta TriLog + Alpha-style Packaging

Date: 2026-08-12
Status: Approved design, pending implementation plan

## 1. Goal

Produce the final competition submission that **guarantees a valid score** (safety first)
while maximizing balanced accuracy (secondary). The model is based on the Theta v4 TriLog
two-tower architecture, trained with more datasets than any previous attempt, and packaged
in the Alpha-style directory layout that avoids the Beta submission's fatal packaging bugs.

Hard constraints from the official PDF (`information_files/B_20260601.pdf`):

- 10 evaluation benchmarks, 10–3000 cases each.
- Any benchmark that fails its runtime limit scores 0 for that benchmark.
- Final score = mean Balanced Accuracy (BA = (TPR + TNR) / 2) over the 10 benchmarks.
- Runtime limits: 30 s (10/30 cases), 100 s (100–3000 cases @ 1M lines), 300 s (100M lines).
- LLM endpoint is optional; `LLM_MODEL_CONFIG` env var (YAML) points to two endpoints
  (oss: nomic-embed-text-v1.5, openai: text-embedding-3-small); higher of the two is used.
- `k` is a soft hint; pairwise BA is used, so bucket count need not equal `k` exactly.

## 2. Datasets (11 total, ~5000 cases)

### Fake pretraining (9 datasets, LODO)
| Dataset | Cases | Bugs |
|---|---:|---:|
| first_batch_dataset | 80 | 8 |
| stage2_dataset_working | 240 | 16 |
| stage3_dataset_32bugs_640cases | 640 | 32 |
| official_vcs_stage1_dataset_v1 | 40 | ? |
| directed_cross_v2 | 37 | ? |
| directed_cross_v4 | 65 | 10 |
| stable_official_like_multitest_v1 | 24 | ? |
| benchmark5_final | 937 | ? |
| benchmark6_final | 2947 | ? |

Note: `stable_official_like_multitest_v1` uses `golden.csv` (not `gold.csv`).

### Official fine-tuning (2 datasets)
| Dataset | Cases | Bugs |
|---|---:|---:|
| benchmark_set_1 | 7 | 2 |
| benchmark_set_2 | 25 | 4 |

## 3. Model Architecture: TriLog Two-Tower

Pair probability network `P(i, j) = same_bug`:

```
Base Tower (256d hidden, LayerNorm/GELU/Dropout + ResidualBlock)
  inputs: LLM features-embedding relation (abs diff + hadamard)
          LLM summary-embedding relation
          multi-view relations (event/object/context)
          structured pair scalars (21-dim)

Trace Tower (256d hidden, same blocks)
  inputs: global trace struct + text embeddings
          anchor trace struct + text embeddings
          residual trace struct + text embeddings

Fusion Head
  concat(base_out, trace_out) -> Linear 512 -> LayerNorm -> ResidualBlock
  -> Linear 256 -> ResidualBlock -> Linear 1 -> sigmoid
```

- Loss: focal loss (γ=2.0), class-balanced pair sampling.
- Base and Trace towers both participate in gradient updates.
- All three log types used: sim.log + regr.log feed the Base Tower; trace.log.gz
  feeds the Trace Tower.

## 4. Two-Stage Training

### Stage 1: 9-dataset LODO pretraining
- 9-fold Leave-One-Dataset-Out over the fake datasets.
- Each fold trains on 8 datasets, holds out 1.
- Pair sampling is episode-local (pairs only within one dataset).
- 5 seeds → 45 pretrained models.

### Stage 2: Official fine-tuning
- Each Stage-1 fold-model is fine-tuned on benchmark_set_1 + benchmark_set_2.
- Freeze backbone; train only last 1–2 layers + head (~5–10% of parameters).
- Higher official pair weight to avoid catastrophic forgetting.
- 5–10 epochs, small learning rate.
- Fine-tuning does not break LODO: a fold-model still never saw its held-out fake dataset.

### GPU
- Max 2 GPUs; must verify they are idle before use; release after use.

## 5. Inference & Packaging

### Directory layout (Alpha-style, fixing Beta bugs)
```
final_submission/
├── regr_fail_bucketing          # shell entry, MUST be at root
├── inference.py                 # Python inference main
├── models/
│   ├── manifest.json            # folds, seeds, dims, clusterer, consensus weights
│   ├── model_{fold}_seed{N}.pt  # 45 TriLog models (fine-tuned)
│   └── preprocess_{fold}_seed{N}.pkl
└── README.md
```

### Shell wrapper (safety first)
```
1. parse --input --output --k
2. write singleton emergency output immediately (every case its own bucket)
3. launch Python inference in background
4. watchdog kills inference at the official runtime limit
5. on success, inference output overwrites singleton
6. on timeout/crash, singleton is preserved
```

### Python inference flow
```
read input.csv -> parse sim/regr/trace logs
  -> LLM embedding via LLM_MODEL_CONFIG (OpenAI-compatible /v1/embeddings)
     - on failure: Base Tower falls back to deterministic features
  -> build pair feature matrix
  -> load 9 fold-models -> predict pair probabilities
  -> fold ensemble (mean) + seed consensus (co-association)
  -> quality-gated agglomerative clustering
  -> write Case,bucket CSV
```

### LLM configuration
- Read `LLM_MODEL_CONFIG` (YAML) per official Section 3.4.
- Use `embedding.config` (base_url, api_key, model) with `openai.AsyncOpenAI`.
- No hardcoded localhost dependency. Local `tools/nomic_openai_embedding_server.py`
  is only for local validation.
- On any LLM failure: degrade to deterministic-only features.

## 6. Clustering Improvement: Quality-Gated Agglomerative

Addresses the chaining problem (A~B, B~C but A≁C get merged wrongly).

- Keep consensus co-association matrix + agglomerative clustering (average linkage).
- Add a deterministic quality gate before each merge:
  - if the representative pair across the two candidate clusters has strong conflict
    (primary_type both nonempty & different, mismatch_type both nonempty & different,
    fatal_file both nonempty & different) → refuse the merge.
- This is deterministic, adds no training cost, and directly prevents false merges.

Further research directions (optional, not required for v1): signed-graph clustering,
correlation clustering (objective aligns with BA), learned bucket-merge scorer.

## 7. Fault Tolerance Chain

From worst to best case, the program always produces a valid `Case,bucket` output:
1. empty input → empty output
2. log read failure → use available logs, missing ones get empty features
3. LLM API failure → Base Tower falls back to deterministic (trace + structured)
4. model load failure → pure deterministic Drain+SVD+Agglomerative
5. any other exception → preserve singleton output

## 8. Validation Matrix

After training, run full LODO evaluation on all 11 datasets and verify:
1. no leakage (each held-out score comes from models that never saw it)
2. official benchmark_set_1/set_2 BA ≥ 0.85 (target)
3. fake datasets (stage2/stage3/VCS) do not regress
4. benchmark5/6 BA and runtime (< 100 s)
5. LLM-failure fallback produces acceptable scores

## 9. Out of Scope (YAGNI)

- End-to-end differentiable clustering (GNN/set-transformer).
- Completion LLM for cluster merging (runtime risk; embedding only).
- FT-Transformer and other backbone alternatives.
- Static hard-positive mining (proven harmful in prior experiments).
