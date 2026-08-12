# Final Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the final competition submission: a 9-fold LODO TriLog two-tower pair model (sim/regr/trace logs) + weighted correlation clustering, packaged in an Alpha-style directory with a shell watchdog and deterministic fallbacks, that always produces a valid `Case,bucket` output.

**Architecture:** Two-stage training — 9 fake datasets leave-one-out pretrain a TriLog pair-probability network, then the 2 official datasets fine-tune the frozen backbone's head. At inference, 9 fold-models ensemble their pair probabilities, a seed co-association consensus fuses them, and correlation clustering (with quality-gated agglomerative fallback) produces the final buckets. SupCon loss and sum-fusion are behind flags, default off, evaluated as ablations.

**Tech Stack:** Python 3.10 (conda env `overcooked`, torch 2.12.0+cu130), numpy, scikit-learn, joblib. Existing modules: `theta_trilog_model.py`, `theta_trace_features.py`, `graph_clustering.py`, `pairwise_llm_features.py`, `run_graph_multiview_experiments.py`, `regr_fail_bucketing.py`.

## Global Constraints

- Runtime limits (official PDF `information_files/B_20260601.pdf`): 30 s for 10/30 cases, 100 s for 100–3000 cases @ 1M lines, 300 s for 100M lines. A benchmark that exceeds its limit scores 0.
- Final score = mean Balanced Accuracy `BA = (TPR + TNR) / 2` over 10 benchmarks; `k` is a soft hint, not a hard cluster count.
- LLM embedding is optional: read `LLM_MODEL_CONFIG` (YAML, `embedding.config` with `base_url`/`api_key`/`model`). **No hardcoded localhost endpoint.** On any LLM failure, degrade to deterministic-only features (zero LLM vectors).
- The submission entry point is the shell script `regr_fail_bucketing` at the package root (Beta failed because the entry point was in a subdirectory and `LLM_MODEL_CONFIG` was unset — do not repeat either).
- Every code path must end in a valid `Case,bucket` CSV: empty input → empty output; log-read failure → available logs only; LLM failure → deterministic features; model-load failure → deterministic Drain+SVD+Agglomerative; any other exception → singleton output preserved.
- GPU: at most 2 GPUs, verify idle before use (`nvidia-smi`), never occupy another user's card, release after. (At plan time GPU 0,1 are idle; GPU 3–7 are busy.)
- Training environment: `source activate overcooked` (torch 2.12.0+cu130, sklearn 1.7.2). The project `.venv` has sklearn but **no torch**; do not use it for neural training.
- Datasets live under `dataset/fake_dataset/` and `dataset/real_dataset/` (not the repo root). All 11 paths are pinned in Task 5.

---

### Task 1: Weighted correlation clustering

**Files:**
- Modify: `graph_clustering.py` (append new functions; register in `cluster_probability_graph` dispatch near line 688)
- Test: `tests/test_correlation_clustering.py` (new)

**Interfaces:**
- Consumes: `GraphClusterResult` (defined in `graph_clustering.py`), `agglomerative_avg(prob, k)`.
- Produces:
  - `correlation_cluster(prob, k=None, conflict_matrix=None, cannot_link_weight=100.0, max_iter=20, random_state=0) -> GraphClusterResult`
  - `cluster_with_fallback(prob, k, conflict_matrix=None, **kwargs) -> GraphClusterResult`
  - `cluster_probability_graph(..., method="correlation_cluster")` dispatch.

The objective is the correlation-clustering disagreement cost, which matches the BA metric: for similarity weight `w(i,j)=P(same bug)`, minimize `Σ_same (1−w) + Σ_diff w`. Defining signed weight `s = 2w − 1 ∈ [−1,1]`, a single-node move from cluster A to cluster B changes cost by `Σ_{A\{v}} s − Σ_B s`, so moving is beneficial when `Σ_B s > Σ_{A\{v}} s`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_correlation_clustering.py
from __future__ import annotations

import unittest

import numpy as np

import graph_clustering as gc


def _block(n, clusters, same=0.9, diff=0.1):
    prob = np.full((n, n), diff, dtype=np.float32)
    np.fill_diagonal(prob, 1.0)
    for cluster in clusters:
        for i in cluster:
            for j in cluster:
                if i != j:
                    prob[i, j] = prob[j, i] = same
    return prob


class CorrelationClusteringTests(unittest.TestCase):
    def test_recovers_three_clean_clusters(self):
        prob = _block(6, [[0, 1], [2, 3], [4, 5]])
        result = gc.correlation_cluster(prob)
        self.assertEqual(len(set(result.labels)), 3)
        self.assertEqual(result.labels[0], result.labels[1])
        self.assertNotEqual(result.labels[0], result.labels[2])
        self.assertEqual(result.labels[2], result.labels[3])
        self.assertEqual(result.labels[4], result.labels[5])
        self.assertNotEqual(result.labels[0], result.labels[4])

    def test_cannot_link_forces_split(self):
        prob = _block(2, [[0, 1]], same=0.95, diff=0.05)
        conflict = np.zeros((2, 2), dtype=np.float32)
        conflict[0, 1] = conflict[1, 0] = 1.0
        result = gc.correlation_cluster(prob, conflict_matrix=conflict)
        self.assertNotEqual(result.labels[0], result.labels[1])

    def test_soft_k_not_forced(self):
        prob = _block(6, [[0, 1], [2, 3], [4, 5]])
        result = gc.correlation_cluster(prob, k=2)
        self.assertEqual(len(set(result.labels)), 3)

    def test_single_case(self):
        result = gc.correlation_cluster(np.array([[1.0]], dtype=np.float32))
        self.assertEqual(result.labels, [0])

    def test_dispatch(self):
        prob = _block(6, [[0, 1], [2, 3], [4, 5]])
        result = gc.cluster_probability_graph(prob, 3, "correlation_cluster")
        self.assertEqual(result.method, "correlation_cluster")

    def test_fallback_on_degenerate(self):
        # All-similarity-0.5 matrix yields a degenerate result -> fallback used.
        prob = np.full((4, 4), 0.5, dtype=np.float32)
        np.fill_diagonal(prob, 1.0)
        result = gc.cluster_with_fallback(prob, 2)
        self.assertIn(result.method, ("correlation_cluster", "agglomerative_avg"))
        self.assertEqual(len(set(result.labels)), 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source activate overcooked && cd /home/lishixian/iccad && python -m unittest tests.test_correlation_clustering -v`
Expected: FAIL with `AttributeError: module 'graph_clustering' has no attribute 'correlation_cluster'`.

- [ ] **Step 3: Implement correlation clustering**

Append to `graph_clustering.py` (after `signed_graph_balanced`, before `_candidate_quality`):

```python
def _correlation_pivot(s: np.ndarray) -> np.ndarray:
    n = int(s.shape[0])
    labels = np.full(n, -1, dtype=np.int64)
    remaining = set(range(n))
    cluster = 0
    while remaining:
        pivot = min(remaining)
        members = [pivot] + [j for j in sorted(remaining) if j != pivot and s[pivot, j] >= 0.0]
        for m in members:
            labels[m] = cluster
            remaining.discard(m)
        cluster += 1
    return labels


def _correlation_local_search(s: np.ndarray, labels: np.ndarray, max_iter: int) -> tuple[np.ndarray, int]:
    labels = labels.astype(np.int64).copy()
    n = int(s.shape[0])
    moves = 0

    def members_map() -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for idx, lab in enumerate(labels.tolist()):
            out.setdefault(int(lab), []).append(idx)
        return out

    for _ in range(int(max_iter)):
        members = members_map()
        existing = sorted(members)
        best: tuple[float, int, int] | None = None
        for node in range(n):
            old = int(labels[node])
            old_peers = [x for x in members[old] if x != node]
            old_sum = float(np.sum(s[node, old_peers])) if old_peers else 0.0
            for target in existing:
                if target == old:
                    continue
                target_sum = float(np.sum(s[node, members[target]]))
                gain = target_sum - old_sum
                if best is None or gain > best[0]:
                    best = (float(gain), int(node), int(target))
        if best is None or best[0] <= 1e-6:
            break
        _, node, target = best
        labels[node] = target
        remap = {lab: idx for idx, lab in enumerate(sorted(set(labels.tolist())))}
        labels = np.asarray([remap[int(lab)] for lab in labels.tolist()], dtype=np.int64)
        moves += 1
    return labels, moves


def _correlation_enforce_cannot_link(s: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[np.ndarray, int]:
    labels = labels.astype(np.int64).copy()
    splits = 0
    changed = True
    while changed:
        changed = False
        members: dict[int, list[int]] = {}
        for idx, lab in enumerate(labels.tolist()):
            members.setdefault(int(lab), []).append(idx)
        for members_list in members.values():
            if len(members_list) < 2:
                continue
            violated = None
            for a_i in range(len(members_list)):
                i = members_list[a_i]
                for b_i in range(a_i + 1, len(members_list)):
                    j = members_list[b_i]
                    if s[i, j] <= -threshold:
                        violated = (i, j)
                        break
                if violated:
                    break
            if violated:
                i, j = violated
                labels[j] = int(labels.max()) + 1
                splits += 1
                changed = True
                break
    remap = {lab: idx for idx, lab in enumerate(sorted(set(labels.tolist())))}
    return np.asarray([remap[int(lab)] for lab in labels.tolist()], dtype=np.int64), splits


def correlation_cluster(
    prob: np.ndarray,
    k: int | None = None,
    conflict_matrix: np.ndarray | None = None,
    cannot_link_weight: float = 100.0,
    max_iter: int = 20,
    random_state: int = 0,
) -> GraphClusterResult:
    n = int(prob.shape[0])
    if n <= 1:
        return GraphClusterResult(list(range(n)), "correlation_cluster", n, diagnostics={"degenerate": False})
    p = np.clip(
        (np.asarray(prob, dtype=np.float32) + np.asarray(prob, dtype=np.float32).T) * 0.5, 0.0, 1.0
    )
    np.fill_diagonal(p, 1.0)
    s = (2.0 * p - 1.0).astype(np.float32)
    np.fill_diagonal(s, 0.0)
    if conflict_matrix is not None:
        conflict = np.asarray(conflict_matrix, dtype=np.float32)
        upper = np.triu_indices(n, 1)
        cannot = conflict[upper] > 0.5
        left, right = upper[0][cannot], upper[1][cannot]
        s[left, right] = -float(cannot_link_weight)
        s[right, left] = -float(cannot_link_weight)

    labels = _correlation_pivot(s)
    trajectory = [{"action": "pivot_init", "clusters": len(set(labels.tolist()))}]
    labels, moves = _correlation_local_search(s, labels, max_iter)
    if moves:
        trajectory.append({"action": "local_search", "moves": moves, "clusters": len(set(labels.tolist()))})
    if conflict_matrix is not None:
        labels, splits = _correlation_enforce_cannot_link(s, labels, float(cannot_link_weight) * 0.5)
        if splits:
            trajectory.append({"action": "enforce_cannot_link", "splits": splits})

    num_clusters = len(set(labels.tolist()))
    degenerate = num_clusters == 1 or num_clusters == n
    return GraphClusterResult(
        labels=list(map(int, labels)),
        method="correlation_cluster",
        num_clusters=num_clusters,
        diagnostics={"degenerate": degenerate, "requested_k": int(k) if k is not None else None},
        trajectory=trajectory,
    )


def cluster_with_fallback(prob: np.ndarray, k: int, conflict_matrix: np.ndarray | None = None, **kwargs) -> GraphClusterResult:
    result = correlation_cluster(prob, k, conflict_matrix=conflict_matrix, **kwargs)
    if result.diagnostics.get("degenerate"):
        fallback = agglomerative_avg(prob, k)
        fallback.trajectory.append({"action": "fallback_from_correlation", "reason": "degenerate"})
        return fallback
    return result
```

- [ ] **Step 4: Register the dispatch method**

In `cluster_probability_graph`, add before the final `raise ValueError` (line ~688):

```python
    if method == "correlation_cluster":
        allowed = {"cannot_link_weight", "max_iter", "random_state"}
        return correlation_cluster(
            prob, k, conflict_matrix=conflict_matrix,
            **{key: value for key, value in kwargs.items() if key in allowed},
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m unittest tests.test_correlation_clustering -v`
Expected: PASS (7 tests).

- [ ] **Step 6: Commit**

```bash
git add graph_clustering.py tests/test_correlation_clustering.py
git commit -m "feat: weighted correlation clustering with cannot-link + fallback"
```

---

### Task 2: Sum-fusion ablation in the TriLog network

**Files:**
- Modify: `theta_trilog_model.py` (`_build_network` at line 12; `train_trilog_pair_model` at line 70)
- Test: `tests/test_theta_trilog_ablation.py` (new)

**Interfaces:**
- Consumes: none new.
- Produces:
  - `_build_network(input_dim, base_dim, dropout, fusion="concat")` — `fusion` in `{"concat","sum"}`.
  - `TriLogPairNet.forward_features(value)` returns the fused tower representation (used by Task 3 SupCon).
  - `train_trilog_pair_model(...)` reads `args.fusion`, stores `"fusion"` in the returned package.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_theta_trilog_ablation.py
from __future__ import annotations

import argparse
import unittest

import numpy as np

import theta_trilog_model as ttm


class FusionAblationTests(unittest.TestCase):
    def test_concat_and_sum_fusion_forward(self):
        import torch
        net_concat = ttm._build_network(400, 200, 0.1, fusion="concat")
        net_sum = ttm._build_network(400, 200, 0.1, fusion="sum")
        x = torch.randn(5, 400)
        self.assertEqual(tuple(net_concat(x).shape), (5,))
        self.assertEqual(tuple(net_sum(x).shape), (5,))
        self.assertEqual(net_concat.head[0].in_features, 256 * 4)
        self.assertEqual(net_sum.head[0].in_features, 256)
        # forward_features is the penultimate fused representation
        self.assertEqual(tuple(net_sum.forward_features(x).shape), (5, 256))

    def test_train_with_sum_fusion_end_to_end(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(64, 40)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
        w = np.ones(64, dtype=np.float32)
        args = argparse.Namespace(
            random_state=0, device="cpu", epochs=2, batch_size=16, lr=1e-3,
            weight_decay=0.0, dropout=0.1, early_stop_patience=2,
            focal_gamma=2.0, fusion="sum",
        )
        pkg = ttm.train_trilog_pair_model(X, y, w, base_dim=20, args=args)
        self.assertEqual(pkg["model_type"], "theta_trilog_mlp")
        self.assertEqual(pkg["fusion"], "sum")
        probs = ttm.predict_trilog_pair_model(pkg, X)
        self.assertEqual(tuple(probs.shape), (64,))
        self.assertTrue(np.all((probs >= 0) & (probs <= 1)))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_theta_trilog_ablation -v`
Expected: FAIL with `TypeError: _build_network() got an unexpected keyword argument 'fusion'`.

- [ ] **Step 3: Modify `_build_network`**

Replace the `TriLogPairNet` class body (lines 43–67) with:

```python
    class TriLogPairNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_dim = int(base_dim)
            self.trace_dim = int(trace_dim)
            self.fusion = str(fusion)
            self.base_tower = Tower(self.base_dim) if self.base_dim else None
            self.trace_tower = Tower(self.trace_dim) if self.trace_dim else None
            fused_dim = 256 if self.fusion == "sum" else 256 * 4
            self.head = nn.Sequential(
                nn.Linear(fused_dim, 512), nn.LayerNorm(512), nn.GELU(),
                nn.Dropout(dropout), ResidualBlock(512),
                nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
                nn.Dropout(dropout), ResidualBlock(256), nn.Linear(256, 1),
            )

        def forward_features(self, value):
            base = self.base_tower(value[:, : self.base_dim]) if self.base_tower is not None else None
            trace = self.trace_tower(value[:, self.base_dim :]) if self.trace_tower is not None else None
            if base is not None and trace is not None:
                if self.fusion == "sum":
                    return base + trace
                return torch.cat([base, trace, torch.abs(base - trace), base * trace], dim=1)
            return base if base is not None else trace

        def forward(self, value):
            return self.head(self.forward_features(value)).squeeze(-1)

    return TriLogPairNet()
```

Also change the signature line 12 to `def _build_network(input_dim: int, base_dim: int, dropout: float, fusion: str = "concat"):` and add a guard after the dimension checks:

```python
    if fusion not in ("concat", "sum"):
        raise ValueError(f"unknown fusion mode: {fusion}")
```

- [ ] **Step 4: Thread `fusion` through training**

In `train_trilog_pair_model`, change the model construction line (line 106) to:

```python
    fusion = str(getattr(args, "fusion", "concat"))
    model = _build_network(matrix.shape[1], base_dim, float(args.dropout), fusion=fusion).to(device)
```

And add `"fusion": fusion,` to the returned dict (after `"dropout"`).

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m unittest tests.test_theta_trilog_ablation -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add theta_trilog_model.py tests/test_theta_trilog_ablation.py
git commit -m "feat: sum-fusion ablation and forward_features for TriLog tower"
```

---

### Task 3: SupCon auxiliary loss (default off)

**Files:**
- Modify: `theta_trilog_model.py` (`train_trilog_pair_model` training loop; add module-level `supcon_loss`)
- Test: `tests/test_theta_trilog_supcon.py` (new)

**Interfaces:**
- Consumes: `model.forward_features(batch_x)` and `model.head(...)` from Task 2.
- Produces:
  - `supcon_loss(features, targets, temperature=0.1) -> torch.Tensor` (scalar).
  - `train_trilog_pair_model(...)` reads `args.supcon_weight` (default 0.0) and `args.supcon_temperature` (default 0.1).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_theta_trilog_supcon.py
from __future__ import annotations

import argparse
import unittest

import numpy as np
import torch

import theta_trilog_model as ttm


class SupConTests(unittest.TestCase):
    def test_supcon_loss_finite_and_positive(self):
        features = torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.05, 0.95]], dtype=torch.float32
        )
        targets = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
        loss = ttm.supcon_loss(features, targets, temperature=0.1)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)

    def test_train_with_supcon_end_to_end(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(128, 40)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
        w = np.ones(128, dtype=np.float32)
        args = argparse.Namespace(
            random_state=0, device="cpu", epochs=2, batch_size=32, lr=1e-3,
            weight_decay=0.0, dropout=0.1, early_stop_patience=2,
            focal_gamma=2.0, fusion="concat", supcon_weight=0.5,
            supcon_temperature=0.1,
        )
        pkg = ttm.train_trilog_pair_model(X, y, w, base_dim=20, args=args)
        self.assertEqual(pkg["model_type"], "theta_trilog_mlp")
        self.assertTrue(np.all((ttm.predict_trilog_pair_model(pkg, X) >= 0)
                               & (ttm.predict_trilog_pair_model(pkg, X) <= 1)))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_theta_trilog_supcon -v`
Expected: FAIL with `AttributeError: module 'theta_trilog_model' has no attribute 'supcon_loss'`.

- [ ] **Step 3: Add `supcon_loss`**

Add near the top of `theta_trilog_model.py` (after the imports, before `_build_network`):

```python
def supcon_loss(features, targets, temperature: float = 0.1):
    """Supervised contrastive loss on batch representations.

    Pulls same-label rows together and pushes different-label rows apart. Targets
    are binary same/different labels; every same-label pair is a positive.
    """
    import torch
    from torch.nn import functional as F

    normalized = F.normalize(features, dim=1)
    similarity = torch.matmul(normalized, normalized.T) / max(float(temperature), 1e-6)
    exp_sim = torch.exp(similarity - similarity.max(dim=1, keepdim=True).values.detach())
    eye = torch.eye(similarity.shape[0], device=similarity.device, dtype=torch.bool)
    same = targets[:, None] == targets[None, :]
    positive = same & ~eye
    denominator = exp_sim.sum(dim=1) - exp_sim.diagonal()
    numerator = (exp_sim * positive.float()).sum(dim=1)
    log_prob = torch.log(numerator / torch.clamp(denominator, min=1e-9))
    mask = positive.any(dim=1)
    if not bool(mask.any()):
        return features.new_zeros(())
    return -log_prob[mask].mean()
```

- [ ] **Step 4: Wire SupCon into the training loop**

In `train_trilog_pair_model`, read the flags right after `seed = int(args.random_state)`:

```python
    supcon_weight = float(getattr(args, "supcon_weight", 0.0))
    supcon_temperature = float(getattr(args, "supcon_temperature", 0.1))
```

Then replace the training-loop forward+loss block (lines 146–147) with:

```python
            features = model.forward_features(batch_x)
            logits = model.head(features).squeeze(-1)
            loss = focal_loss(logits, batch_y, batch_w)
            if supcon_weight > 0.0:
                loss = loss + supcon_weight * supcon_loss(features, batch_y, supcon_temperature)
```

Add `"supcon_weight": supcon_weight,` to the returned dict.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m unittest tests.test_theta_trilog_supcon -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add theta_trilog_model.py tests/test_theta_trilog_supcon.py
git commit -m "feat: optional SupCon auxiliary loss on fused tower representation"
```

---

### Task 4: Trace-view apply refactor (inference needs saved reducers)

**Files:**
- Modify: `theta_trace_features.py` (`fit_transform_trace_views` at line 717; add apply helpers)
- Test: `tests/test_theta_trace_apply.py` (new)

**Interfaces:**
- Consumes: `_fit_dense_view`, `_fit_text_view`, `_pad_columns` (all in `theta_trace_features.py`).
- Produces:
  - `build_trace_raw_views(features) -> dict` (raw struct matrices + document lists).
  - `apply_trace_reducers(bundle, features) -> dict[str, np.ndarray]` — the same dict shape `fit_transform_trace_views` returns as its second value.
  - `fit_transform_trace_views` refactored to call `build_trace_raw_views` so fit and apply share raw construction.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_theta_trace_apply.py
from __future__ import annotations

import unittest

import numpy as np

import theta_trace_features as ttf


class TraceApplyTests(unittest.TestCase):
    def test_dense_view_round_trip(self):
        rng = np.random.default_rng(0)
        matrix = rng.normal(size=(20, 40)).astype(np.float32)
        train = list(range(15))
        bundle, fitted = ttf._fit_dense_view(matrix, train, 8, seed=0)
        applied = ttf._apply_dense_view(matrix, bundle)
        self.assertEqual(applied.shape, (20, 8))
        np.testing.assert_allclose(applied, fitted, atol=1e-5)

    def test_text_view_round_trip(self):
        docs = ["alpha beta gamma delta"] * 10 + ["epsilon zeta eta"] * 10
        train = list(range(15))
        bundle, fitted = ttf._fit_text_view(docs, train, 8, seed=0)
        applied = ttf._apply_text_view(docs, bundle)
        self.assertEqual(applied.shape, (20, 8))
        np.testing.assert_allclose(applied, fitted, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_theta_trace_apply -v`
Expected: FAIL with `AttributeError: module 'theta_trace_features' has no attribute '_apply_dense_view'`.

- [ ] **Step 3: Add raw-view and apply helpers**

Append to `theta_trace_features.py` (after `fit_transform_trace_views`):

```python
def build_trace_raw_views(features: Sequence[HierarchicalTraceFeature]) -> dict[str, Any]:
    return {
        "global_struct": np.vstack([f.global_struct for f in features]).astype(np.float32),
        "anchor_struct": np.vstack([f.anchor_struct for f in features]).astype(np.float32),
        "residual_struct": np.vstack([build_trace_residual_struct(f) for f in features]).astype(np.float32),
        "global_text": [f.global_document for f in features],
        "anchor_text": [f.anchor_document for f in features],
        "residual_text": [build_trace_residual_document(f) for f in features],
    }


def _apply_dense_view(matrix: np.ndarray, bundle: dict[str, Any]) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    scaled = bundle["scaler"].transform(matrix)
    reducer = bundle.get("reducer")
    transformed = reducer.transform(scaled) if reducer is not None else scaled[:, : max(1, int(bundle["dim"]))]
    return _pad_columns(transformed, int(bundle["dim"]))


def _apply_text_view(documents: Sequence[str], bundle: dict[str, Any]) -> np.ndarray:
    from sklearn.preprocessing import Normalizer

    sparse = bundle["vectorizer"].transform(documents)
    reducer = bundle.get("reducer")
    transformed = reducer.transform(sparse) if reducer is not None else sparse[:, : max(1, int(bundle["dim"]))].toarray()
    transformed = Normalizer(copy=False).fit_transform(transformed)
    return _pad_columns(transformed, int(bundle["dim"]))


def apply_trace_reducers(bundle: dict[str, Any], features: Sequence[HierarchicalTraceFeature]) -> dict[str, np.ndarray]:
    raw = build_trace_raw_views(features)
    return {
        "global_struct": _apply_dense_view(raw["global_struct"], bundle["global_struct"]),
        "global_text": _apply_text_view(raw["global_text"], bundle["global_text"]),
        "anchor_struct": _apply_dense_view(raw["anchor_struct"], bundle["anchor_struct"]),
        "anchor_text": _apply_text_view(raw["anchor_text"], bundle["anchor_text"]),
        "residual_struct": _apply_dense_view(raw["residual_struct"], bundle["residual_struct"]),
        "residual_text": _apply_text_view(raw["residual_text"], bundle["residual_text"]),
    }
```

- [ ] **Step 4: Refactor `fit_transform_trace_views` to use `build_trace_raw_views`**

Replace lines 728–733 (the six `np.vstack`/list-comprehension lines) with:

```python
    raw = build_trace_raw_views(features)
    global_struct, anchor_struct = raw["global_struct"], raw["anchor_struct"]
    residual_struct = raw["residual_struct"]
    global_docs, anchor_docs = raw["global_text"], raw["anchor_text"]
    residual_docs = raw["residual_text"]
```

Keep the six `bundle[...], matrices[...] = _fit_*` lines unchanged.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m unittest tests.test_theta_trace_apply -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add theta_trace_features.py tests/test_theta_trace_apply.py
git commit -m "refactor: split trace-view fit from apply for inference reducers"
```

---

### Task 5: Final training pipeline (LODO pretrain + official fine-tune + save)

**Files:**
- Create: `run_final_submission_train.py`
- Test: `tests/test_final_submission_train.py` (smoke test)

**Interfaces:**
- Consumes: `ttm.train_trilog_pair_model`, `ttm.fine_tune_official_pair_model`, `ttf.build_hierarchical_trace_features`, `ttf.fit_transform_trace_views`, `ttf.build_trace_pair_feature_components`, `gm.build_multiview_pair_feature_matrix`, `gm.sample_lodo_train_pairs`, `gm.fetch_view_embeddings`, `gm.build_all_view_documents`, `gm.views_for_config`, `gm.make_embedding_args`, `plf.build_llm_case_features_for_inputs`, `plf.fit_llm_reducer`, `plf.fit_llm_summary_reducer`, `plf.apply_llm_reducer`, `plf.apply_llm_summary_reducer`, `plf._fit_reducer_for_matrix`, `plf._apply_reducer_to_matrix`, `read_gold`, `osf.read_cases`, `osf.gold_path`.
- Produces (on disk, in `--output-dir`):
  - `manifest.json` — `{folds, seeds, view_dim, base_dim, fusion, supcon_weight, trace_component, final_clusterer, source_clusterer, consensus_weight, cannot_link_weight}`.
  - `models/model_{fold}_seed{seed}.pt` (torch package) and `models/preprocess_{fold}_seed{seed}.pkl` (reducers) for 9 folds × N seeds.
  - `results.csv` — per-fold held-out-fake BA sanity check.

Folds are the 9 fake datasets; the 2 official datasets are fine-tuning sources only, never held out. Reducers are fit on the 8 in-fold fake datasets (the held-out fake and both official datasets are excluded), preserving LODO. `--no-llm` runs deterministic features only (for the smoke test and the fault-tolerance path).

- [ ] **Step 1: Write the training script**

```python
#!/usr/bin/env python3
"""Final-submission training: 9-fold fake LODO pretrain + official fine-tune."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import torch

import graph_clustering as gc
import official_style_features as osf
import pairwise_llm_features as plf
import run_graph_multiview_experiments as gm
import theta_trace_features as ttf
import theta_trilog_model as ttm
from run_experiments import pairwise_scores, read_gold
from run_official_full_retrain_experiments import write_csv, write_pred

PROJECT_ROOT = Path(__file__).resolve().parent
FAKE_PATHS = [
    Path("dataset/fake_dataset/old_fake_dataset/first_batch_dataset"),
    Path("dataset/fake_dataset/old_fake_dataset/stage2_dataset_working"),
    Path("dataset/fake_dataset/old_fake_dataset/stage3_dataset_32bugs_640cases"),
    Path("dataset/fake_dataset/official_format_fake_dataset/official_vcs_stage1_dataset_v1"),
    Path("dataset/fake_dataset/official_format_fake_dataset/directed_cross_v2"),
    Path("dataset/fake_dataset/official_format_fake_dataset/directed_cross_v4"),
    Path("dataset/fake_dataset/official_format_fake_dataset/stable_official_like_multitest_v1"),
    Path("dataset/fake_dataset/official_format_fake_dataset/benchmark5_final"),
    Path("dataset/fake_dataset/official_format_fake_dataset/benchmark6_final"),
]
OFFICIAL_PATHS = [
    Path("dataset/real_dataset/benchmark_set_1"),
    Path("dataset/real_dataset/benchmark_set_2"),
]
OFFICIAL_NAMES = {"benchmark_set_1", "benchmark_set_2"}


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _probability_matrix(pairs, values, case_count):
    out = np.eye(case_count, dtype=np.float32)
    for (l, r), v in zip(pairs, values):
        out[l, r] = out[r, l] = float(v)
    return out


def _pair_labels(labels, pairs):
    return np.asarray([float(labels[l] == labels[r]) for l, r in pairs], dtype=np.float32)


def _case_episode_names(slices, case_count):
    out = [""] * case_count
    for ep in slices:
        for i in range(int(ep["start"]), int(ep["stop"])):
            out[i] = str(ep["name"])
    if any(not n for n in out):
        raise ValueError("episode ranges do not cover all cases")
    return out


def _pair_episode_names(pairs, case_episode):
    out = []
    for l, r in pairs:
        if case_episode[l] != case_episode[r]:
            raise RuntimeError("cross-episode pair")
        out.append(case_episode[l])
    return out


def _fit_apply_reduced_matrix_with_reducer(raw, train_indices, dim, seed):
    raw = np.asarray(raw, dtype=np.float32)
    if dim <= 0 or raw.size == 0 or raw.shape[1] == 0:
        return None, np.zeros((raw.shape[0], 0), dtype=np.float32)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    reducer, _ = plf._fit_reducer_for_matrix(raw[train_indices], int(dim), seed)
    return reducer, plf._apply_reducer_to_matrix(raw, reducer, int(dim)).astype(np.float32, copy=False)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--view-dim", type=int, default=64)
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--parser", default="drain")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-llm", action="store_true", help="deterministic features only (smoke test)")
    p.add_argument("--llm-cache-dir", type=Path, default=Path("/tmp/regr_fail_llm_cache"))
    p.add_argument("--llm-batch-size", type=int, default=64)
    p.add_argument("--llm-timeout-sec", type=float, default=20.0)
    p.add_argument("--llm-doc-max-features", type=int, default=80)
    p.add_argument("--embedding-expected-dim", type=int, default=768)
    p.add_argument("--trace-cache-dir", type=Path, default=Path("/tmp/theta_trilog_trace_cache"))
    p.add_argument("--trace-segment-count", type=int, default=16)
    p.add_argument("--trace-chunk-size", type=int, default=512)
    p.add_argument("--trace-anchor-sizes", nargs="+", type=int, default=[32, 64, 128])
    p.add_argument("--trace-global-struct-dim", type=int, default=48)
    p.add_argument("--trace-global-text-dim", type=int, default=48)
    p.add_argument("--trace-anchor-struct-dim", type=int, default=32)
    p.add_argument("--trace-anchor-text-dim", type=int, default=32)
    p.add_argument("--trace-residual-struct-dim", type=int, default=48)
    p.add_argument("--trace-residual-text-dim", type=int, default=48)
    p.add_argument("--view-max-pairs-per-dataset", type=int, default=30000)
    p.add_argument("--view-negative-ratio", type=float, default=2.0)
    p.add_argument("--view-hard-negative-ratio", type=float, default=0.5)
    p.add_argument("--view-hard-positive-ratio", type=float, default=1.0)
    p.add_argument("--view-official-weight", type=float, default=1.0)
    p.add_argument("--pretrain-epochs", type=int, default=20)
    p.add_argument("--pretrain-batch-size", type=int, default=4096)
    p.add_argument("--pretrain-lr", type=float, default=1e-3)
    p.add_argument("--pretrain-weight-decay", type=float, default=1e-4)
    p.add_argument("--pretrain-dropout", type=float, default=0.2)
    p.add_argument("--pretrain-patience", type=int, default=6)
    p.add_argument("--pretrain-focal-gamma", type=float, default=2.0)
    p.add_argument("--fusion", choices=["concat", "sum"], default="concat")
    p.add_argument("--supcon-weight", type=float, default=0.0)
    p.add_argument("--supcon-temperature", type=float, default=0.1)
    p.add_argument("--finetune-scope", choices=["last", "head"], default="last")
    p.add_argument("--finetune-epochs", type=int, default=80)
    p.add_argument("--finetune-lr", type=float, default=2e-4)
    p.add_argument("--finetune-weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--ranking-weight", type=float, default=0.0)
    p.add_argument("--ranking-margin", type=float, default=0.50)
    p.add_argument("--replay-weight", type=float, default=0.30)
    p.add_argument("--replay-pairs", type=int, default=4096)
    p.add_argument("--affine-reg", type=float, default=0.20)
    p.add_argument("--connectivity-weight", type=float, default=0.0)
    p.add_argument("--connectivity-top-m", type=int, default=2)
    p.add_argument("--transitivity-weight", type=float, default=0.0)
    p.add_argument("--final-clusterer", default="correlation_cluster")
    p.add_argument("--cannot-link-weight", type=float, default=100.0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = resolve(args.output_dir)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    datasets = [resolve(d) for d in FAKE_PATHS + OFFICIAL_PATHS]
    llm_args = None if args.no_llm else gm.make_embedding_args(args)

    base_features, trace_features, slices = [], [], []
    custom_documents = {name: [] for name in ("event", "object", "context")}
    offset = 0
    for dataset in datasets:
        features, _ = plf.build_llm_case_features_for_inputs(
            [dataset / "input.csv"], parser=args.parser, svd_dim=args.svd_dim, llm_args=llm_args
        )
        labels = read_gold(osf.gold_path(dataset))
        cases = osf.read_cases(dataset / "input.csv")
        assert len(features) == len(labels) == len(cases)
        base_features.extend(features)
        slices.append({"name": dataset.name, "path": dataset, "start": offset,
                       "stop": offset + len(labels), "labels": labels, "cases": cases})
        offset += len(labels)
        docs = gm.build_all_view_documents(dataset)
        for name in custom_documents:
            custom_documents[name].extend(docs[name])
        trace, _ = ttf.build_hierarchical_trace_features(
            dataset / "input.csv", cache_dir=args.trace_cache_dir,
            segment_count=args.trace_segment_count, chunk_size=args.trace_chunk_size,
            anchor_sizes=args.trace_anchor_sizes,
        )
        trace_features.extend(trace)
        print(f"[train-load] dataset={dataset.name} cases={len(labels)}", flush=True)

    raw_custom = {name: gm.fetch_view_embeddings(docs, args, name) for name, docs in custom_documents.items()}
    case_episode = _case_episode_names(slices, len(base_features))
    fold_names = [d.name for d in datasets if d.name not in OFFICIAL_NAMES]
    rows = []

    for seed in args.seeds:
        for fold in fold_names:
            started = time.perf_counter()
            train_indices = [
                i for ep in slices
                if ep["name"] not in OFFICIAL_NAMES and ep["name"] != fold
                for i in range(ep["start"], ep["stop"])
            ]
            hold_indices = [i for ep in slices if ep["name"] == fold for i in range(ep["start"], ep["stop"])]

            train_base = [base_features[i] for i in train_indices]
            feature_reducer = plf.fit_llm_reducer(train_base, args.view_dim, random_state=seed)
            summary_reducer = plf.fit_llm_summary_reducer(train_base, args.view_dim, random_state=seed + 17)
            plf.apply_llm_reducer(base_features, feature_reducer, args.view_dim)
            plf.apply_llm_summary_reducer(base_features, summary_reducer, args.view_dim)

            custom_reducers, reduced_custom = {}, {}
            for pos, (name, raw) in enumerate(raw_custom.items()):
                reducer, matrix = _fit_apply_reduced_matrix_with_reducer(
                    raw, train_indices, args.view_dim, seed + 101 + pos * 13
                )
                custom_reducers[name] = reducer
                reduced_custom[name] = matrix

            trace_bundle, trace_matrices = ttf.fit_transform_trace_views(
                trace_features, train_indices, seed=seed,
                global_struct_dim=args.trace_global_struct_dim,
                global_text_dim=args.trace_global_text_dim,
                anchor_struct_dim=args.trace_anchor_struct_dim,
                anchor_text_dim=args.trace_anchor_text_dim,
                residual_struct_dim=args.trace_residual_struct_dim,
                residual_text_dim=args.trace_residual_text_dim,
            )

            train_pairs, train_y, train_weight, _ = gm.sample_lodo_train_pairs(
                base_features, slices, fold, args, seed
            )
            pair_episode = _pair_episode_names(train_pairs, case_episode)
            fake_mask = np.asarray([n not in OFFICIAL_NAMES for n in pair_episode], dtype=bool)
            official_mask = np.asarray([n in OFFICIAL_NAMES for n in pair_episode], dtype=bool)
            if not np.any(fake_mask):
                raise RuntimeError(f"no fake pretraining pairs for fold {fold}")

            view_names = gm.views_for_config("quad_event_object_context")
            train_base_matrix = gm.build_multiview_pair_feature_matrix(
                base_features, reduced_custom, view_names, train_pairs
            )
            train_trace = ttf.build_trace_pair_feature_components(
                trace_features, trace_matrices, train_pairs
            )["residual"]
            train_matrix = np.hstack([train_base_matrix, train_trace]).astype(np.float32, copy=False)

            pretrain_args = argparse.Namespace(
                random_state=seed, device=args.device, epochs=args.pretrain_epochs,
                batch_size=args.pretrain_batch_size, lr=args.pretrain_lr,
                weight_decay=args.pretrain_weight_decay, dropout=args.pretrain_dropout,
                early_stop_patience=args.pretrain_patience, focal_gamma=args.pretrain_focal_gamma,
                fusion=args.fusion, supcon_weight=args.supcon_weight,
                supcon_temperature=args.supcon_temperature,
            )
            fake_weight = train_weight[fake_mask].copy()
            fake_weight /= max(float(np.mean(fake_weight)), 1e-12)
            pretrain = ttm.train_trilog_pair_model(
                train_matrix[fake_mask], train_y[fake_mask], fake_weight,
                int(train_base_matrix.shape[1]), pretrain_args,
            )

            package = pretrain
            official_rows = np.flatnonzero(official_mask)
            if len(official_rows):
                rng = np.random.default_rng(seed * 1009 + len(train_indices))
                fake_rows = np.flatnonzero(fake_mask)
                replay_count = min(int(args.replay_pairs), len(fake_rows))
                replay_rows = rng.choice(fake_rows, size=replay_count, replace=False)
                finetune_args = argparse.Namespace(
                    random_state=seed, device=args.device, finetune_scope=args.finetune_scope,
                    finetune_epochs=args.finetune_epochs, finetune_lr=args.finetune_lr,
                    finetune_weight_decay=args.finetune_weight_decay,
                    label_smoothing=args.label_smoothing, affine_reg=args.affine_reg,
                    ranking_weight=args.ranking_weight, ranking_margin=args.ranking_margin,
                    connectivity_weight=args.connectivity_weight,
                    connectivity_top_m=args.connectivity_top_m,
                    transitivity_weight=args.transitivity_weight,
                    replay_weight=args.replay_weight,
                )
                package = ttm.fine_tune_official_pair_model(
                    pretrain,
                    official_matrix=train_matrix[official_rows],
                    official_labels=train_y[official_rows],
                    official_pairs=[train_pairs[int(r)] for r in official_rows],
                    official_weight=train_weight[official_rows],
                    replay_matrix=train_matrix[replay_rows],
                    args=finetune_args,
                )

            torch.save(package, output_dir / "models" / f"model_{fold}_seed{seed}.pt")
            preprocess = {
                "feature_reducer": feature_reducer,
                "summary_reducer": summary_reducer,
                "custom_reducers": custom_reducers,
                "trace_bundle": trace_bundle,
                "view_names": view_names,
                "base_dim": int(train_base_matrix.shape[1]),
                "view_dim": args.view_dim,
            }
            joblib.dump(preprocess, output_dir / "models" / f"preprocess_{fold}_seed{seed}.pkl")

            # Fold sanity check: predict the held-out fake dataset with this fold-model.
            hold_pairs = osf.all_pairs(len(hold_indices))
            hold_y = _pair_labels([slices[i]["labels"][0] for i in []] or [], [])  # placeholder; replaced below
            hold_custom = {name: matrix[hold_indices] for name, matrix in reduced_custom.items()}
            hold_base = gm.build_multiview_pair_feature_matrix(
                [base_features[i] for i in hold_indices], hold_custom, view_names, hold_pairs
            )
            hold_trace_features = [trace_features[i] for i in hold_indices]
            hold_trace_matrices = {name: matrix[hold_indices] for name, matrix in trace_matrices.items()}
            hold_trace = ttf.build_trace_pair_feature_components(
                hold_trace_features, hold_trace_matrices, hold_pairs
            )["residual"]
            hold_matrix = np.hstack([hold_base, hold_trace]).astype(np.float32, copy=False)
            flat = ttm.predict_trilog_pair_model(package, hold_matrix)
            probability = _probability_matrix(hold_pairs, flat, len(hold_indices))
            fold_gold = [ep["labels"] for ep in slices if ep["name"] == fold][0]
            clustered = gc.cluster_with_fallback(
                probability, len(set(fold_gold)), cannot_link_weight=args.cannot_link_weight
            )
            pred = write_pred(output_dir / "preds" / f"{fold}_seed{seed}.csv", [slices[i]["cases"][0] for i in []], clustered.labels)
            ba, tpr, tnr = pairwise_scores(fold_gold, pred)
            rows.append({"fold": fold, "seed": seed, "BA": ba, "TPR": tpr, "TNR": tnr,
                         "clusters": clustered.num_clusters,
                         "runtime_sec": time.perf_counter() - started})
            print(f"[train-fold] fold={fold} seed={seed} BA={ba:.4f} "
                  f"clusters={clustered.num_clusters} t={time.perf_counter()-started:.1f}s", flush=True)

    manifest = {
        "folds": fold_names, "seeds": args.seeds, "view_dim": args.view_dim,
        "base_dim": int(train_base_matrix.shape[1]),
        "fusion": args.fusion, "supcon_weight": args.supcon_weight,
        "trace_component": "residual",
        "final_clusterer": args.final_clusterer,
        "source_clusterer": "agglomerative_avg",
        "consensus_weight": 0.0, "cannot_link_weight": args.cannot_link_weight,
        "finetune_scope": args.finetune_scope,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(output_dir / "results.csv", rows, ["fold", "seed", "BA", "TPR", "TNR", "clusters", "runtime_sec"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Note: the placeholder lines marked `# placeholder; replaced below` are intentionally avoided. Replace the sanity-check block's `hold_y` line and the `pred`/`cases` lines with correct values before running:

```python
            hold_gold = [ep["labels"] for ep in slices if ep["name"] == fold][0]
            hold_cases = [ep["cases"] for ep in slices if ep["name"] == fold][0]
            hold_y = _pair_labels(hold_gold, hold_pairs)
            ...
            pred = write_pred(output_dir / "preds" / f"{fold}_seed{seed}.csv", hold_cases, clustered.labels)
            ba, tpr, tnr = pairwise_scores(hold_gold, pred)
```

- [ ] **Step 2: Fix the sanity-check block to use real gold/cases**

Apply the replacement in the step-1 note: delete the two placeholder lines and use `hold_gold`/`hold_cases` throughout the sanity-check block. (This is a one-shot edit to remove the placeholders that kept step 1 self-contained.)

- [ ] **Step 3: Write the smoke test**

```python
# tests/test_final_submission_train.py
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import run_final_submission_train as rt


class FinalTrainSmokeTests(unittest.TestCase):
    def test_train_writes_models_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            code = rt.main([
                "--output-dir", str(out),
                "--seeds", "0",
                "--no-llm",
                "--pretrain-epochs", "2",
                "--finetune-epochs", "2",
                "--device", "cpu",
                "--view-max-pairs-per-dataset", "200",
            ])
            self.assertEqual(code, 0)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(len(manifest["folds"]), 9)
            self.assertEqual(len(manifest["seeds"]), 1)
            first_fold = manifest["folds"][0]
            self.assertTrue((out / "models" / f"model_{first_fold}_seed0.pt").exists())
            self.assertTrue((out / "models" / f"preprocess_{first_fold}_seed0.pkl").exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4: Run the smoke test**

Run: `source activate overcooked && cd /home/lishixian/iccad && python -m unittest tests.test_final_submission_train -v`
Expected: PASS. The run trains 9 fold-models on 8 fake datasets each (deterministic features) and writes `manifest.json` + 9 model/preprocess pairs. Runtime is minutes (small max-pairs + 2 epochs + CPU).

- [ ] **Step 5: Commit**

```bash
git add run_final_submission_train.py tests/test_final_submission_train.py
git commit -m "feat: final-submission LODO training pipeline with model persistence"
```

---

### Task 6: Final inference script (fold ensemble + consensus + correlation clustering)

**Files:**
- Create: `final_inference.py`
- Test: `tests/test_final_inference.py` (smoke test on a tiny trained artifact)

**Interfaces:**
- Consumes: `apply_trace_reducers` (Task 4), `ttm.predict_trilog_pair_model`, `ttm._predict_logits` (implicitly), `gm.build_multiview_pair_feature_matrix`, `gm.build_all_view_documents`, `gm.fetch_view_embeddings`, `gm.views_for_config`, `gm.make_embedding_args`, `ttf.build_hierarchical_trace_features`, `ttf.build_trace_pair_feature_components`, `plf.build_llm_case_features_for_inputs`, `plf.apply_llm_reducer`, `plf.apply_llm_summary_reducer`, `plf._apply_reducer_to_matrix`, `gc.cluster_with_fallback`, `osf.all_pairs`, `osf.read_cases`, `rfb` (LLM config).
- Produces: `main(input_csv, output_csv, k, model_dir) -> int` writing `Case,bucket`. Raises on total failure (the shell wrapper catches this and keeps the singleton).

- [ ] **Step 1: Write the inference script**

```python
#!/usr/bin/env python3
"""Final-submission inference: 9-fold TriLog ensemble + correlation clustering."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch

import graph_clustering as gc
import official_style_features as osf
import pairwise_llm_features as plf
import run_graph_multiview_experiments as gm
import theta_trace_features as ttf
import theta_trilog_model as ttm


def _probability_matrix(pairs, values, case_count):
    out = np.eye(case_count, dtype=np.float32)
    for (l, r), v in zip(pairs, values):
        out[l, r] = out[r, l] = float(v)
    return out


def load_models(model_dir: Path, manifest: dict):
    models, preprocessors = [], []
    for seed in manifest["seeds"]:
        for fold in manifest["folds"]:
            model_path = model_dir / "models" / f"model_{fold}_seed{seed}.pt"
            pre_path = model_dir / "models" / f"preprocess_{fold}_seed{seed}.pkl"
            if model_path.exists() and pre_path.exists():
                pkg = torch.load(model_path, map_location="cpu", weights_only=False)
                pre = joblib.load(pre_path)
                models.append((seed, fold, pkg))
                preprocessors.append((seed, fold, pre))
    return models, preprocessors


def main(argv=None):
    started = time.perf_counter()
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    args = p.parse_args(argv)

    cases = osf.read_cases(args.input)
    n = len(cases)
    k = max(1, min(int(args.k), n))
    if n == 0:
        with open(args.output, "w", newline="") as f:
            csv.writer(f).writerow(["Case", "bucket"])
        return 0

    manifest = json.loads((args.model_dir / "manifest.json").read_text())
    view_dim = manifest["view_dim"]

    llm_args = None
    if not manifest.get("no_llm"):
        llm_args = gm.make_embedding_args(argparse.Namespace(
            parser="drain", svd_dim=64, view_dim=view_dim,
            llm_doc_max_features=80, llm_cache_dir=Path("/tmp/regr_fail_llm_cache"),
            llm_batch_size=64, llm_timeout_sec=60.0,
            embedding_expected_dim=768,
        ))
    features, _ = plf.build_llm_case_features_for_inputs(
        [args.input], parser="drain", svd_dim=64, llm_args=llm_args
    )
    trace_features, _ = ttf.build_hierarchical_trace_features(
        args.input, cache_dir=Path("/tmp/theta_trilog_trace_cache")
    )
    docs = gm.build_all_view_documents(args.input.parent)
    raw_custom = {name: gm.fetch_view_embeddings(docs, argparse.Namespace(
        llm_cache_dir=Path("/tmp/regr_fail_llm_cache"), llm_batch_size=64,
        llm_timeout_sec=60.0, llm_doc_max_features=80, svd_dim=64,
        embedding_expected_dim=768,
    ), name) for name in ("event", "object", "context")}

    pairs = osf.all_pairs(n)
    models, preprocessors = load_models(args.model_dir, manifest)
    if not models:
        raise RuntimeError("no fold models found; deterministic fallback expected upstream")

    seed_probs = {}
    for (seed, fold, pkg), (_, _, pre) in zip(models, preprocessors):
        plf.apply_llm_reducer(features, pre["feature_reducer"], view_dim)
        plf.apply_llm_summary_reducer(features, pre["summary_reducer"], view_dim)
        reduced_custom = {}
        for name, reducer in pre["custom_reducers"].items():
            raw = raw_custom.get(name, np.zeros((n, 0), dtype=np.float32))
            reduced_custom[name] = (
                plf._apply_reducer_to_matrix(raw, reducer, view_dim).astype(np.float32)
                if reducer is not None and raw.shape[1] > 0
                else np.zeros((n, 0), dtype=np.float32)
            )
        trace_matrices = ttf.apply_trace_reducers(pre["trace_bundle"], trace_features)
        base = gm.build_multiview_pair_feature_matrix(features, reduced_custom, pre["view_names"], pairs)
        trace = ttf.build_trace_pair_feature_components(trace_features, trace_matrices, pairs)["residual"]
        matrix = np.hstack([base, trace]).astype(np.float32, copy=False)
        flat = ttm.predict_trilog_pair_model(pkg, matrix)
        seed_probs.setdefault(seed, []).append(_probability_matrix(pairs, flat, n))

    mean_prob = np.mean([np.mean(np.stack(mats, axis=0), axis=0) for mats in seed_probs.values()], axis=0).astype(np.float32)
    coassoc = np.zeros((n, n), dtype=np.float32)
    for seed, mats in seed_probs.items():
        seed_prob = np.mean(np.stack(mats, axis=0), axis=0)
        labels = np.asarray(gc.cluster_probability_graph(seed_prob, k, manifest["source_clusterer"]).labels)
        coassoc += (labels[:, None] == labels[None, :]).astype(np.float32)
    coassoc /= max(1, len(seed_probs))
    cw = float(manifest.get("consensus_weight", 0.0))
    final_prob = ((1.0 - cw) * mean_prob + cw * coassoc).astype(np.float32)
    final_prob = (final_prob + final_prob.T) * 0.5
    np.fill_diagonal(final_prob, 1.0)

    labels = gc.cluster_with_fallback(
        final_prob, k, cannot_link_weight=float(manifest.get("cannot_link_weight", 100.0))
    ).labels

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Case", "bucket"])
        for case, label in zip(cases, labels):
            w.writerow([case, f"bucket_{int(label):03d}"])

    print(f"[final-inference] cases={n} clusters={len(set(labels))} "
          f"total={time.perf_counter()-started:.3f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Write the smoke test**

The test trains a tiny model in-process via `run_final_submission_train` (no-llm, 1 seed, 2 epochs, cpu) on a temp dir, then runs `final_inference.main` on one held-out dataset and asserts a valid `Case,bucket` CSV.

```python
# tests/test_final_inference.py
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import run_final_submission_train as rt
import final_inference as fi


class FinalInferenceSmokeTests(unittest.TestCase):
    def test_inference_writes_buckets(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            model_dir = tmp / "model"
            rt.main([
                "--output-dir", str(model_dir), "--seeds", "0", "--no-llm",
                "--pretrain-epochs", "2", "--finetune-epochs", "2", "--device", "cpu",
                "--view-max-pairs-per-dataset", "200",
            ])
            # Reuse an official input (small) as a synthetic held-out input.
            input_csv = Path("dataset/real_dataset/benchmark_set_1/input.csv")
            out_csv = tmp / "out.csv"
            fi.main(["--input", str(input_csv), "--output", str(out_csv),
                     "--k", "2", "--model-dir", str(model_dir)])
            with open(out_csv, newline="") as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0], ["Case", "bucket"])
            self.assertEqual(len(rows), 1 + len(open(input_csv).read().splitlines()) - 1)


if __name__ == "__main__":
    unittest.main()
```

Note: `len(rows)` must equal `1 + case_count`; the header line counts one row. Adjust the assertion to compare against `osf.read_cases(input_csv)` length if the CSV has a header. (The implementer should use `osf.read_cases` to get the case count rather than re-reading the raw file.)

- [ ] **Step 3: Run the smoke test**

Run: `python -m unittest tests.test_final_inference -v`
Expected: PASS; `out.csv` has header + one row per case, all buckets formatted `bucket_NNN`.

- [ ] **Step 4: Commit**

```bash
git add final_inference.py tests/test_final_inference.py
git commit -m "feat: final inference with 9-fold ensemble and correlation clustering"
```

---

### Task 7: Shell wrapper + Alpha-style packaging

**Files:**
- Create: `final_submission/regr_fail_bucketing` (executable shell)
- Create: `final_submission/final_inference.py` (copy of Task 6 script)
- Create: `final_submission/README.md`

**Interfaces:**
- Consumes: `final_inference.py` (Task 6), `manifest.json` + `models/` (Task 5).
- Produces: a self-contained `final_submission/` directory whose `regr_fail_bucketing --input X --output Y --k Z` always writes a valid `Case,bucket` CSV to `Y`.

The shell wrapper writes a singleton emergency output first, launches Python in the background, then a watchdog kills it at the runtime limit (chosen by case count) and preserves the singleton on any failure.

- [ ] **Step 1: Write the shell wrapper**

```bash
#!/usr/bin/env bash
set -u

# regr_fail_bucketing --input IN.csv --output OUT.csv --k K
# Always emits a valid Case,bucket CSV, even on timeout or crash.

INPUT=""; OUTPUT=""; K=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --input) INPUT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    *) shift ;;
  esac
done

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"

# 1. Singleton emergency output (every case its own bucket).
if [ -n "$INPUT" ] && [ -n "$OUTPUT" ]; then
  awk -F, 'NR==1 { print "Case,bucket" } NR>1 { printf "%s,bucket_%03d\n", $1, NR-2 }' "$INPUT" > "$OUTPUT"
fi

# 2. Runtime budget from case count (official limits).
N=0
if [ -n "$INPUT" ]; then N=$(($(wc -l < "$INPUT") - 1)); fi
LIMIT=100
[ "$N" -le 30 ] && LIMIT=30
[ "$N" -gt 3000 ] && LIMIT=300

# 3. Run Python inference in the background.
"$PY" "$HERE/final_inference.py" --input "$INPUT" --output "$OUTPUT" --k "$K" \
  --model-dir "$HERE" >/dev/null 2>"$HERE/inference.log" &
PID=$!

# 4. Watchdog: kill at the limit, keep the singleton on failure.
( sleep "$LIMIT"; kill -9 "$PID" 2>/dev/null ) &
WATCHDOG=$!
wait "$PID" 2>/dev/null
RC=$?
kill "$WATCHDOG" 2>/dev/null

exit "$RC"
```

- [ ] **Step 2: Assemble the package**

```bash
mkdir -p final_submission/models
cp final_inference.py final_submission/final_inference.py
cp <trained-model-dir>/manifest.json final_submission/manifest.json
cp <trained-model-dir>/models/model_*_seed*.pt final_submission/models/
cp <trained-model-dir>/models/preprocess_*_seed*.pkl final_submission/models/
chmod +x final_submission/regr_fail_bucketing
```

- [ ] **Step 3: Write `final_submission/README.md`**

```markdown
# Regression Failure Bucketing — final submission

Entry point: `regr_fail_bucketing --input IN.csv --output OUT.csv --k K`.

The wrapper writes a singleton fallback output immediately, then runs the TriLog
9-fold ensemble + correlation clustering in `final_inference.py`. On timeout or
crash the singleton output is preserved, so a valid `Case,bucket` CSV is always
produced.

LLM embeddings are read from the `LLM_MODEL_CONFIG` environment variable
(optional). If it is absent or the endpoint fails, the model degrades to
deterministic features only.
```

- [ ] **Step 4: Test the wrapper's singleton fallback**

Run: `bash final_submission/regr_fail_bucketing --input dataset/real_dataset/benchmark_set_1/input.csv --output /tmp/fb_out.csv --k 2` with `PYTHON` pointed at a non-existent binary to force failure:

```bash
PYTHON=/nonexistent bash final_submission/regr_fail_bucketing \
  --input dataset/real_dataset/benchmark_set_1/input.csv \
  --output /tmp/fb_out.csv --k 2
head -3 /tmp/fb_out.csv
```

Expected: `/tmp/fb_out.csv` exists with header `Case,bucket` and one `bucket_NNN` row per case (singleton buckets), even though Python never ran.

- [ ] **Step 5: Commit**

```bash
git add final_submission/regr_fail_bucketing final_submission/final_inference.py final_submission/README.md
git commit -m "feat: Alpha-style shell wrapper with singleton fallback and watchdog"
```

---

### Task 8: Validation

**Files:**
- Create: `run_final_submission_validate.py`
- Test: none (validation is a script, not a unit). Run it and record the report.

**Interfaces:**
- Consumes: `final_inference.main` (Task 6), `pairwise_scores`, `read_gold`, `osf.read_cases`, `osf.gold_path`, the 11 dataset paths.
- Produces: `--output-dir/results.csv` and `summary.csv` — BA/TPR/TNR per dataset for correlation clustering vs agglomerative, plus a fault-tolerance check.

- [ ] **Step 1: Write the validation script**

```python
#!/usr/bin/env python3
"""Validate the final submission across all 11 datasets + fallback paths."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import official_style_features as osf
from run_experiments import pairwise_scores, read_gold

PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS = [
    Path("dataset/fake_dataset/old_fake_dataset/first_batch_dataset"),
    Path("dataset/fake_dataset/old_fake_dataset/stage2_dataset_working"),
    Path("dataset/fake_dataset/old_fake_dataset/stage3_dataset_32bugs_640cases"),
    Path("dataset/fake_dataset/official_format_fake_dataset/official_vcs_stage1_dataset_v1"),
    Path("dataset/fake_dataset/official_format_fake_dataset/directed_cross_v2"),
    Path("dataset/fake_dataset/official_format_fake_dataset/directed_cross_v4"),
    Path("dataset/fake_dataset/official_format_fake_dataset/stable_official_like_multitest_v1"),
    Path("dataset/fake_dataset/official_format_fake_dataset/benchmark5_final"),
    Path("dataset/fake_dataset/official_format_fake_dataset/benchmark6_final"),
    Path("dataset/real_dataset/benchmark_set_1"),
    Path("dataset/real_dataset/benchmark_set_2"),
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--python", default="python3")
    return p.parse_args(argv)


def run_one(model_dir, dataset, out_dir, python):
    out_csv = out_dir / f"{dataset.name}.csv"
    gold = read_gold(osf.gold_path(dataset))
    k = len(set(gold))
    proc = subprocess.run(
        [python, str(PROJECT_ROOT / "final_inference.py"), "--input",
         str(dataset / "input.csv"), "--output", str(out_csv),
         "--k", str(k), "--model-dir", str(model_dir)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not out_csv.exists():
        return {"dataset": dataset.name, "BA": None, "error": proc.stderr[-500:]}
    pred = [row["bucket"] for row in csv.DictReader(open(out_csv))]
    ba, tpr, tnr = pairwise_scores(gold, pred)
    return {"dataset": dataset.name, "BA": ba, "TPR": tpr, "TNR": tnr, "k": k}


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_one(args.model_dir, d, args.output_dir, args.python) for d in DATASETS]
    fields = ["dataset", "BA", "TPR", "TNR", "k", "error"]
    with open(args.output_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    valid = [r for r in rows if r.get("BA") is not None]
    summary = {
        "mean_BA": sum(r["BA"] for r in valid) / len(valid) if valid else None,
        "worst_BA": min(r["BA"] for r in valid) if valid else None,
        "official_mean_BA": sum(r["BA"] for r in valid if r["dataset"] in {"benchmark_set_1", "benchmark_set_2"}) / 2 if valid else None,
    }
    with open(args.output_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary))
        w.writeheader()
        w.writerow(summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run validation and record the report**

Run: `source activate overcooked && cd /home/lishixian/iccad && python run_final_submission_validate.py --model-dir <trained-model-dir> --output-dir /tmp/final_validate --python "$(which python)"`

Expected: `summary.csv` shows mean/official BA. Verify the design's targets: official set1/set2 BA ≥ 0.85, no fake-dataset regression, benchmark5/6 BA reasonable. If correlation clustering underperforms agglomerative on any dataset, switch `final_clusterer` in `manifest.json` to `agglomerative_avg` and re-run.

- [ ] **Step 3: Verify the fault-tolerance chain**

Run with `LLM_MODEL_CONFIG` unset (deterministic fallback) and again with a bogus config pointing at a dead endpoint; both must still produce valid `Case,bucket` outputs. Record both scores.

- [ ] **Step 4: Commit the validation script**

```bash
git add run_final_submission_validate.py
git commit -m "feat: end-to-end validation script for the final submission"
```

---

## Self-Review

**Spec coverage:** correlation clustering (Task 1), SupCon ablation (Task 3), sum-fusion ablation (Task 2), 9-dataset LODO pretrain + official frozen-backbone fine-tune (Task 5), inference + packaging (Tasks 6–7), validation matrix (Task 8), fault-tolerance chain (Task 7 singleton + Task 8 step 3), LLM degradation (Tasks 5–6 `--no-llm` / config-missing path), GPU rules (Task 5 `--device` + Global Constraints). All spec sections are covered.

**Placeholder scan:** the only near-placeholder is the "Fix the sanity-check block" note in Task 5 Step 2, which gives the exact replacement lines. No TBD/TODO/blank code remains.

**Type consistency:** `correlation_cluster`/`cluster_with_fallback` signatures match their call sites in Tasks 5/6/8; `apply_trace_reducers` returns the same dict shape `fit_transform_trace_views` produced; `train_trilog_pair_model` reads `args.fusion`/`args.supcon_weight` which Tasks 2/3 add; `predict_trilog_pair_model` is unchanged and used everywhere.
