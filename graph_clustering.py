#!/usr/bin/env python3
"""Experimental graph-aware clustering for pair probability matrices.

This module is intentionally independent from the formal predictor.  It only
consumes a P_same matrix, optional structured conflict matrix, and the requested
reference k.  Gold labels are never used here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


@dataclass
class GraphClusterResult:
    labels: list[int]
    method: str
    num_clusters: int
    num_merges: int = 0
    num_splits: int = 0
    runtime_sec: float = 0.0
    diagnostics: dict = field(default_factory=dict)
    trajectory: list[dict] = field(default_factory=list)


def dense_labels_from_clusters(clusters: Sequence[Sequence[int]], n: int) -> list[int]:
    labels = [-1] * n
    for cluster_id, members in enumerate(clusters):
        for idx in members:
            labels[int(idx)] = cluster_id
    # Singletons for any missing items. This should not normally trigger, but
    # keeps experimental postprocesses robust.
    next_label = len(clusters)
    for idx, label in enumerate(labels):
        if label < 0:
            labels[idx] = next_label
            next_label += 1
    return labels


def clusters_from_labels(labels: Sequence[int]) -> list[list[int]]:
    groups: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        groups.setdefault(int(label), []).append(idx)
    return [groups[key] for key in sorted(groups)]


def _agglomerative(prob: np.ndarray, k: int, linkage: str) -> list[int]:
    from sklearn.cluster import AgglomerativeClustering

    n = int(prob.shape[0])
    if n == 0:
        return []
    k = max(1, min(int(k), n))
    if k == n:
        return list(range(n))
    dist = 1.0 - np.clip(np.asarray(prob, dtype=np.float32), 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    try:
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage=linkage)
    except TypeError:
        model = AgglomerativeClustering(n_clusters=k, affinity="precomputed", linkage=linkage)
    return model.fit_predict(dist).astype(int).tolist()


def agglomerative_avg(prob: np.ndarray, k: int) -> GraphClusterResult:
    labels = _agglomerative(prob, k, "average")
    return GraphClusterResult(labels=labels, method="agglomerative_avg", num_clusters=len(set(labels)))


def agglomerative_complete(prob: np.ndarray, k: int) -> GraphClusterResult:
    labels = _agglomerative(prob, k, "complete")
    return GraphClusterResult(labels=labels, method="agglomerative_complete", num_clusters=len(set(labels)))


def _cluster_internal_quality(prob: np.ndarray, members: Sequence[int], top_m: int = 5) -> float:
    members = list(members)
    if len(members) <= 1:
        return 1.0
    values = prob[np.ix_(members, members)]
    tri = values[np.triu_indices(len(members), k=1)]
    if tri.size == 0:
        return 1.0
    m = min(max(1, int(top_m)), tri.size)
    top = np.partition(tri, tri.size - m)[-m:] if tri.size > m else tri
    return float(np.mean(top))


def _cross_stats(
    prob: np.ndarray,
    conflict: np.ndarray,
    left: Sequence[int],
    right: Sequence[int],
    top_m: int,
) -> dict:
    values = np.asarray(prob[np.ix_(left, right)], dtype=np.float32).reshape(-1)
    conflicts = np.asarray(conflict[np.ix_(left, right)], dtype=np.float32).reshape(-1)
    if values.size == 0:
        return {
            "top_m_cross_mean": 0.0,
            "cross_mean": 0.0,
            "cross_min": 0.0,
            "conflict_ratio": 0.0,
        }
    m = min(max(1, int(top_m)), values.size)
    top = np.partition(values, values.size - m)[-m:] if values.size > m else values
    return {
        "top_m_cross_mean": float(np.mean(top)),
        "cross_mean": float(np.mean(values)),
        "cross_min": float(np.min(values)),
        "conflict_ratio": float(np.mean(conflicts > 0.5)) if conflicts.size else 0.0,
    }


def conservative_merge(
    prob: np.ndarray,
    k: int,
    conflict_matrix: np.ndarray | None = None,
    merge_top_m: int = 5,
    merge_threshold: float = 0.75,
    merge_conflict_threshold: float = 0.20,
    merge_internal_threshold: float = 0.55,
    merge_max_merges: int = 2,
) -> GraphClusterResult:
    n = int(prob.shape[0])
    if n <= 1:
        return GraphClusterResult(list(range(n)), "conservative_merge", n)
    conflict = np.zeros_like(prob, dtype=np.float32) if conflict_matrix is None else np.asarray(conflict_matrix, dtype=np.float32)
    labels = _agglomerative(prob, k, "average")
    clusters = clusters_from_labels(labels)
    trajectory: list[dict] = []
    merges = 0
    while merges < int(merge_max_merges) and len(clusters) > 1:
        best: tuple[float, int, int, dict] | None = None
        internal = [_cluster_internal_quality(prob, members, merge_top_m) for members in clusters]
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                stats = _cross_stats(prob, conflict, clusters[i], clusters[j], merge_top_m)
                min_internal = min(internal[i], internal[j])
                score = (
                    0.50 * stats["top_m_cross_mean"]
                    + 0.20 * stats["cross_mean"]
                    + 0.20 * min_internal
                    - 0.40 * stats["conflict_ratio"]
                )
                candidate_stats = dict(stats)
                candidate_stats.update({
                    "merge_score": float(score),
                    "min_cluster_internal_quality": float(min_internal),
                    "left_size": len(clusters[i]),
                    "right_size": len(clusters[j]),
                })
                if best is None or score > best[0]:
                    best = (float(score), i, j, candidate_stats)
        if best is None:
            break
        score, i, j, stats = best
        if (
            stats["top_m_cross_mean"] < float(merge_threshold)
            or stats["conflict_ratio"] > float(merge_conflict_threshold)
            or stats["min_cluster_internal_quality"] < float(merge_internal_threshold)
        ):
            break
        merged = sorted(clusters[i] + clusters[j])
        before = len(clusters)
        clusters = [c for idx, c in enumerate(clusters) if idx not in {i, j}] + [merged]
        clusters.sort(key=lambda x: (x[0], len(x)))
        merges += 1
        stats.update({"clusters_before": before, "clusters_after": len(clusters)})
        trajectory.append(stats)
    labels = dense_labels_from_clusters(clusters, n)
    return GraphClusterResult(
        labels=labels,
        method="conservative_merge",
        num_clusters=len(clusters),
        num_merges=merges,
        trajectory=trajectory,
    )


def _connected_components(n: int, edges: Iterable[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in edges:
        union(int(i), int(j))
    groups: dict[int, list[int]] = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)
    return [groups[key] for key in sorted(groups)]


def _merge_clusters_to_k(prob: np.ndarray, clusters: list[list[int]], k: int) -> tuple[list[list[int]], int, list[dict]]:
    merges = 0
    trajectory: list[dict] = []
    while len(clusters) > k:
        best: tuple[float, int, int] | None = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                score = float(np.mean(prob[np.ix_(clusters[i], clusters[j])]))
                if best is None or score > best[0]:
                    best = (score, i, j)
        if best is None:
            break
        score, i, j = best
        before = len(clusters)
        merged = sorted(clusters[i] + clusters[j])
        clusters = [c for idx, c in enumerate(clusters) if idx not in {i, j}] + [merged]
        clusters.sort(key=lambda x: (x[0], len(x)))
        merges += 1
        trajectory.append({
            "action": "merge_to_k",
            "score": float(score),
            "clusters_before": before,
            "clusters_after": len(clusters),
        })
    return clusters, merges, trajectory


def _split_cluster_once(prob: np.ndarray, members: Sequence[int]) -> tuple[list[int], list[int]] | None:
    members = list(members)
    if len(members) <= 1:
        return None
    if len(members) == 2:
        return [members[0]], [members[1]]
    sub_prob = prob[np.ix_(members, members)]
    sub_labels = _agglomerative(sub_prob, 2, "complete")
    left = [members[idx] for idx, label in enumerate(sub_labels) if int(label) == 0]
    right = [members[idx] for idx, label in enumerate(sub_labels) if int(label) != 0]
    if not left or not right:
        return None
    return sorted(left), sorted(right)


def _split_clusters_to_k(prob: np.ndarray, clusters: list[list[int]], k: int) -> tuple[list[list[int]], int, list[dict]]:
    splits = 0
    trajectory: list[dict] = []
    while len(clusters) < k:
        candidates = [
            (_cluster_internal_quality(prob, members, top_m=max(1, len(members))), idx)
            for idx, members in enumerate(clusters)
            if len(members) > 1
        ]
        if not candidates:
            break
        _quality, idx = min(candidates, key=lambda x: x[0])
        split = _split_cluster_once(prob, clusters[idx])
        if split is None:
            break
        before = len(clusters)
        left, right = split
        old_size = len(clusters[idx])
        clusters = [c for cidx, c in enumerate(clusters) if cidx != idx] + [left, right]
        clusters.sort(key=lambda x: (x[0], len(x)))
        splits += 1
        trajectory.append({
            "action": "split_to_k",
            "clusters_before": before,
            "clusters_after": len(clusters),
            "old_size": old_size,
            "left_size": len(left),
            "right_size": len(right),
        })
    return clusters, splits, trajectory


def mutual_knn_cc(
    prob: np.ndarray,
    k: int,
    conflict_matrix: np.ndarray | None = None,
    mknn_k: int = 5,
    mknn_threshold: float = 0.65,
) -> GraphClusterResult:
    n = int(prob.shape[0])
    if n <= 1:
        return GraphClusterResult(list(range(n)), "mutual_knn_cc", n)
    requested_k = max(1, min(int(k), n))
    conflict = np.zeros_like(prob, dtype=np.float32) if conflict_matrix is None else np.asarray(conflict_matrix, dtype=np.float32)
    p = np.asarray(prob, dtype=np.float32)
    top_neighbors: list[set[int]] = []
    kk = max(1, min(int(mknn_k), n - 1))
    for i in range(n):
        row = p[i].copy()
        row[i] = -np.inf
        top = np.argpartition(row, -kk)[-kk:]
        top_neighbors.append({int(j) for j in top if row[j] >= float(mknn_threshold)})
    edges = []
    for i in range(n):
        for j in top_neighbors[i]:
            if i < j and i in top_neighbors[j] and conflict[i, j] <= 0.5:
                edges.append((i, j))
    clusters = _connected_components(n, edges)
    trajectory: list[dict] = [{
        "action": "mknn_components",
        "edges": len(edges),
        "components": len(clusters),
        "mknn_k": kk,
        "mknn_threshold": float(mknn_threshold),
    }]
    merges = splits = 0
    if len(clusters) > requested_k:
        clusters, merges, extra = _merge_clusters_to_k(p, clusters, requested_k)
        trajectory.extend(extra)
    elif len(clusters) < requested_k:
        clusters, splits, extra = _split_clusters_to_k(p, clusters, requested_k)
        trajectory.extend(extra)
    labels = dense_labels_from_clusters(clusters, n)
    return GraphClusterResult(
        labels=labels,
        method="mutual_knn_cc",
        num_clusters=len(clusters),
        num_merges=merges,
        num_splits=splits,
        diagnostics={"mknn_edges": len(edges)},
        trajectory=trajectory,
    )


def _graph_objective(labels: Sequence[int], logits: np.ndarray) -> float:
    labels = list(map(int, labels))
    score = 0.0
    n = len(labels)
    for i in range(n):
        li = labels[i]
        for j in range(i + 1, n):
            if li == labels[j]:
                score += float(logits[i, j])
    return score


def signed_graph_greedy(
    prob: np.ndarray,
    k: int,
    conflict_matrix: np.ndarray | None = None,
    signed_conflict_penalty: float = 1.0,
    signed_max_iter: int = 20,
    signed_keep_k: bool = True,
) -> GraphClusterResult:
    n = int(prob.shape[0])
    if n <= 1:
        return GraphClusterResult(list(range(n)), "signed_graph_greedy", n)
    requested_k = max(1, min(int(k), n))
    labels = _agglomerative(prob, requested_k, "average")
    conflict = np.zeros_like(prob, dtype=np.float32) if conflict_matrix is None else np.asarray(conflict_matrix, dtype=np.float32)
    p = np.clip(np.asarray(prob, dtype=np.float32), 1e-5, 1.0 - 1e-5)
    logits = np.log(p / (1.0 - p)) - float(signed_conflict_penalty) * conflict
    np.fill_diagonal(logits, 0.0)
    trajectory: list[dict] = []

    def build_members(current_labels: Sequence[int]) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for idx, label in enumerate(current_labels):
            out.setdefault(int(label), []).append(idx)
        return out

    current = _graph_objective(labels, logits)
    for iteration in range(int(signed_max_iter)):
        members = build_members(labels)
        existing = sorted(members)
        best: tuple[float, int, int] | None = None
        for node in range(n):
            old = int(labels[node])
            old_members = members[old]
            if signed_keep_k and len(old_members) <= 1:
                continue
            old_gain = float(np.sum(logits[node, old_members]))
            # self edge is zero, so old_gain is exactly the contribution removed.
            for target in existing:
                if target == old:
                    continue
                target_gain = float(np.sum(logits[node, members[target]]))
                gain = target_gain - old_gain
                if best is None or gain > best[0]:
                    best = (float(gain), int(node), int(target))
        if best is None or best[0] <= 1e-8:
            break
        gain, node, target = best
        old = int(labels[node])
        labels[node] = target
        remap = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        labels = [remap[int(label)] for label in labels]
        current += gain
        trajectory.append({
            "action": "move",
            "iteration": iteration,
            "node": int(node),
            "old_label": int(old),
            "new_label": int(target),
            "gain": float(gain),
            "objective": float(current),
        })
    return GraphClusterResult(
        labels=labels,
        method="signed_graph_greedy",
        num_clusters=len(set(labels)),
        trajectory=trajectory,
    )


def cluster_probability_graph(
    prob: np.ndarray,
    k: int,
    method: str = "agglomerative_avg",
    conflict_matrix: np.ndarray | None = None,
    **kwargs,
) -> GraphClusterResult:
    method = str(method)
    if method == "agglomerative_avg":
        return agglomerative_avg(prob, k)
    if method == "agglomerative_complete":
        return agglomerative_complete(prob, k)
    if method == "conservative_merge":
        allowed = {
            "merge_top_m",
            "merge_threshold",
            "merge_conflict_threshold",
            "merge_internal_threshold",
            "merge_max_merges",
        }
        return conservative_merge(
            prob,
            k,
            conflict_matrix=conflict_matrix,
            **{key: value for key, value in kwargs.items() if key in allowed},
        )
    if method == "mutual_knn_cc":
        allowed = {"mknn_k", "mknn_threshold"}
        return mutual_knn_cc(
            prob,
            k,
            conflict_matrix=conflict_matrix,
            **{key: value for key, value in kwargs.items() if key in allowed},
        )
    if method == "signed_graph_greedy":
        allowed = {"signed_conflict_penalty", "signed_max_iter", "signed_keep_k"}
        return signed_graph_greedy(
            prob,
            k,
            conflict_matrix=conflict_matrix,
            **{key: value for key, value in kwargs.items() if key in allowed},
        )
    raise ValueError(f"unknown graph cluster method: {method}")

