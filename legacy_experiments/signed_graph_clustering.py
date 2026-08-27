#!/usr/bin/env python3
"""Experimental fixed/adaptive clustering for pair-probability graphs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class ClusterResult:
    labels: list[int]
    selected_k: int
    trajectory: list[dict]


def _dense_labels(clusters: Sequence[Sequence[int]], n: int) -> list[int]:
    labels = [-1] * n
    for label, members in enumerate(clusters):
        for idx in members:
            labels[idx] = label
    return labels


def average_cluster(prob: np.ndarray, k: int) -> ClusterResult:
    from pairwise_llm_features import cluster_from_probability

    labels = cluster_from_probability(prob, max(1, min(int(k), len(prob))))
    return ClusterResult(list(map(int, labels)), len(set(labels)), [])


def spectral_cluster(prob: np.ndarray, k: int, random_state: int = 0) -> ClusterResult:
    from sklearn.cluster import SpectralClustering

    n = len(prob)
    k = max(1, min(int(k), n))
    if k == 1:
        return ClusterResult([0] * n, 1, [])
    affinity = np.clip((prob + prob.T) * 0.5, 0.0, 1.0)
    np.fill_diagonal(affinity, 1.0)
    labels = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels="cluster_qr",
        random_state=random_state,
    ).fit_predict(affinity)
    return ClusterResult(labels.astype(int).tolist(), k, [])


def _merge_gain(
    left: Sequence[int],
    right: Sequence[int],
    logits: np.ndarray,
    conflicts: np.ndarray,
    topk: int,
    conflict_penalty: float,
) -> tuple[float, float, float]:
    positive = logits[np.ix_(left, right)].reshape(-1)
    conflict = conflicts[np.ix_(left, right)].reshape(-1)
    k = min(max(1, int(topk)), len(positive))
    top = np.partition(positive, len(positive) - k)[-k:] if len(positive) > k else positive
    positive_score = float(np.mean(top))
    conflict_score = float(np.mean(conflict)) if len(conflict) else 0.0
    return positive_score - conflict_penalty * conflict_score, positive_score, conflict_score


def signed_graph_cluster(
    prob: np.ndarray,
    requested_k: int,
    conflict_matrix: np.ndarray | None = None,
    k_policy: str = "adaptive",
    min_factor: float = 0.8,
    max_factor: float = 1.2,
    topk: int = 10,
    conflict_penalty: float = 2.0,
    cliff_z_min: float = 1.0,
) -> ClusterResult:
    n = int(prob.shape[0])
    if n <= 1:
        return ClusterResult(list(range(n)), n, [])
    requested_k = max(1, min(int(requested_k), n))
    lower = max(1, min(requested_k, int(math.floor(requested_k * min_factor))))
    upper = min(n, max(requested_k, int(math.ceil(requested_k * max_factor))))
    conflicts = (
        np.asarray(conflict_matrix, dtype=np.float32)
        if conflict_matrix is not None else np.zeros_like(prob, dtype=np.float32)
    )
    p = np.clip(np.asarray(prob, dtype=np.float32), 1e-5, 1.0 - 1e-5)
    logits = np.log(p / (1.0 - p))
    np.fill_diagonal(logits, 0.0)
    clusters: list[list[int]] = [[idx] for idx in range(n)]
    partitions: dict[int, list[list[int]]] = {n: [x[:] for x in clusters]}
    trajectory: list[dict] = []
    while len(clusters) > lower:
        best: tuple[float, float, float, int, int] | None = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                gain, positive, conflict = _merge_gain(
                    clusters[i], clusters[j], logits, conflicts, topk, conflict_penalty
                )
                candidate = (gain, positive, -conflict, i, j)
                if best is None or candidate > best:
                    best = candidate
        assert best is not None
        gain, positive, neg_conflict, left_idx, right_idx = best
        left = clusters[left_idx]
        right = clusters[right_idx]
        before = len(clusters)
        trajectory.append({
            "clusters_before": before,
            "clusters_after": before - 1,
            "merge_gain": float(gain),
            "positive_score": float(positive),
            "conflict_score": float(-neg_conflict),
            "left_size": len(left),
            "right_size": len(right),
        })
        merged = sorted(left + right)
        clusters = [
            cluster for idx, cluster in enumerate(clusters)
            if idx not in {left_idx, right_idx}
        ] + [merged]
        clusters.sort(key=lambda members: (members[0], len(members)))
        partitions[len(clusters)] = [x[:] for x in clusters]

    selected_k = requested_k
    if k_policy == "adaptive":
        candidates = [
            row for row in trajectory
            if lower < int(row["clusters_before"]) <= upper
        ]
        if len(candidates) >= 2:
            gains = np.asarray([float(row["merge_gain"]) for row in candidates], dtype=np.float32)
            drops = gains[:-1] - gains[1:]
            scale = float(np.std(gains))
            if scale > 1e-8:
                z = drops / scale
                best_z = float(np.max(z))
                if best_z >= float(cliff_z_min):
                    positions = np.flatnonzero(np.isclose(z, best_z))
                    selected_k = max(int(candidates[int(pos) + 1]["clusters_before"]) for pos in positions)
    selected_k = max(lower, min(upper, selected_k))
    chosen = partitions.get(selected_k, partitions[min(partitions, key=lambda value: abs(value - selected_k))])
    labels = _dense_labels(chosen, n)
    return ClusterResult(labels, len(chosen), trajectory)


def cluster_probability(
    prob: np.ndarray,
    requested_k: int,
    method: str,
    k_policy: str,
    conflict_matrix: np.ndarray | None = None,
    random_state: int = 0,
) -> ClusterResult:
    if method == "average":
        return average_cluster(prob, requested_k)
    if method == "spectral":
        return spectral_cluster(prob, requested_k, random_state)
    if method == "signed_graph":
        return signed_graph_cluster(
            prob,
            requested_k,
            conflict_matrix=conflict_matrix,
            k_policy=k_policy,
        )
    raise ValueError(f"unknown cluster method: {method}")
