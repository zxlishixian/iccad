#!/usr/bin/env python3
"""Fixed-k sparse signed-graph clustering for the experimental Theta route."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class ThetaClusterResult:
    labels: list[int]
    selected_k: int
    method: str
    iterations: int = 0
    moves: int = 0
    objective: float = 0.0
    trajectory: list[dict] = field(default_factory=list)


def _dense_labels(labels: Sequence[int]) -> list[int]:
    mapping = {label: idx for idx, label in enumerate(sorted(set(map(int, labels))))}
    return [mapping[int(label)] for label in labels]


def _repair_exact_k(labels: Sequence[int], k: int, case_matrix: np.ndarray) -> list[int]:
    labels = _dense_labels(labels)
    n = len(labels)
    while len(set(labels)) < k:
        groups: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            groups.setdefault(label, []).append(idx)
        source, members = max(groups.items(), key=lambda item: (len(item[1]), -item[0]))
        if len(members) <= 1:
            break
        sub = case_matrix[members]
        center = np.mean(sub, axis=0)
        distances = np.linalg.norm(sub - center, axis=1)
        split_idx = members[int(np.argmax(distances))]
        labels[split_idx] = max(labels) + 1
        labels = _dense_labels(labels)
    while len(set(labels)) > k:
        groups = {}
        for idx, label in enumerate(labels):
            groups.setdefault(label, []).append(idx)
        source, members = min(groups.items(), key=lambda item: (len(item[1]), item[0]))
        other = [label for label in groups if label != source]
        center = np.mean(case_matrix[members], axis=0)
        target = max(
            other,
            key=lambda label: float(np.dot(center, np.mean(case_matrix[groups[label]], axis=0))),
        )
        labels = [target if label == source else label for label in labels]
        labels = _dense_labels(labels)
    return labels


def _initial_labels(case_matrix: np.ndarray, k: int, random_state: int) -> list[int]:
    from sklearn.cluster import KMeans

    n = len(case_matrix)
    k = max(1, min(int(k), n))
    if k == 1:
        return [0] * n
    if not case_matrix.size or not np.any(case_matrix):
        return [idx % k for idx in range(n)]
    labels = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=5,
        max_iter=100,
        random_state=int(random_state),
    ).fit_predict(case_matrix)
    return _repair_exact_k(labels.tolist(), k, case_matrix)


def _edge_weights(
    probabilities: np.ndarray,
    conflicts: np.ndarray,
    conflict_penalty: float,
) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=np.float32), 1e-4, 1.0 - 1e-4)
    logit = np.log(p / (1.0 - p))
    weights = logit - float(conflict_penalty) * np.asarray(conflicts, dtype=np.float32)
    return np.clip(weights, -8.0, 8.0).astype(np.float32)


def _objective(
    labels: Sequence[int],
    pairs: Sequence[tuple[int, int]],
    weights: np.ndarray,
    k: int,
    balance_weight: float,
) -> float:
    score = sum(
        float(weight)
        for (i, j), weight in zip(pairs, weights)
        if labels[i] == labels[j]
    )
    counts = np.bincount(np.asarray(labels, dtype=np.int32), minlength=k).astype(np.float32)
    target = len(labels) / max(1, k)
    score -= float(balance_weight) * float(np.sum((counts - target) ** 2) / max(1.0, target))
    return float(score)


def sparse_signed_graph_cluster(
    case_matrix: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    probabilities: Sequence[float],
    k: int,
    conflicts: Sequence[float] | None = None,
    conflict_penalty: float = 2.0,
    balance_weight: float = 0.05,
    max_iter: int = 12,
    random_state: int = 0,
) -> ThetaClusterResult:
    matrix = np.asarray(case_matrix, dtype=np.float32)
    n = int(matrix.shape[0])
    if n == 0:
        return ThetaClusterResult([], 0, "theta_signed_graph")
    k = max(1, min(int(k), n))
    if n == 1 or k == n:
        labels = list(range(n)) if k == n else [0]
        return ThetaClusterResult(labels, k, "theta_signed_graph")
    if len(pairs) != len(probabilities):
        raise ValueError("Theta graph pairs/probabilities length mismatch")
    conflict_array = (
        np.zeros(len(pairs), dtype=np.float32)
        if conflicts is None else np.asarray(conflicts, dtype=np.float32)
    )
    if len(conflict_array) != len(pairs):
        raise ValueError("Theta graph conflicts length mismatch")
    weights = _edge_weights(np.asarray(probabilities), conflict_array, conflict_penalty)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for (i, j), weight in zip(pairs, weights):
        adjacency[int(i)].append((int(j), float(weight)))
        adjacency[int(j)].append((int(i), float(weight)))

    labels = _initial_labels(matrix, k, random_state)
    counts = np.bincount(np.asarray(labels), minlength=k).astype(np.int32)
    target_size = n / max(1, k)
    trajectory: list[dict] = []
    moves = 0
    for iteration in range(max(0, int(max_iter))):
        changed = 0
        for node in range(n):
            source = int(labels[node])
            if counts[source] <= 1:
                continue
            edge_scores = np.zeros(k, dtype=np.float64)
            edge_support = np.zeros(k, dtype=np.int32)
            for other, weight in adjacency[node]:
                label = int(labels[other])
                edge_scores[label] += weight
                edge_support[label] += 1
            current = edge_scores[source]
            best_gain = 0.0
            best_target = source
            for target in range(k):
                if target == source:
                    continue
                # Known support prevents a single bridge edge from dominating.
                support_scale = min(1.0, edge_support[target] / 3.0)
                edge_gain = support_scale * edge_scores[target] - current
                before = (counts[source] - target_size) ** 2 + (counts[target] - target_size) ** 2
                after = (counts[source] - 1 - target_size) ** 2 + (counts[target] + 1 - target_size) ** 2
                balance_gain = -float(balance_weight) * (after - before) / max(1.0, target_size)
                gain = float(edge_gain + balance_gain)
                if gain > best_gain + 1e-9:
                    best_gain = gain
                    best_target = target
            if best_target != source:
                labels[node] = best_target
                counts[source] -= 1
                counts[best_target] += 1
                moves += 1
                changed += 1
        trajectory.append({
            "iteration": iteration,
            "moves": changed,
            "cluster_sizes": ";".join(map(str, counts.tolist())),
        })
        if changed == 0:
            break
    labels = _repair_exact_k(labels, k, matrix)
    objective = _objective(labels, pairs, weights, k, balance_weight)
    return ThetaClusterResult(
        labels=labels,
        selected_k=len(set(labels)),
        method="theta_signed_graph",
        iterations=len(trajectory),
        moves=moves,
        objective=objective,
        trajectory=trajectory,
    )


def dense_probability_matrix(
    n: int,
    pairs: Sequence[tuple[int, int]],
    probabilities: Sequence[float],
    missing_probability: float = 0.05,
) -> np.ndarray:
    matrix = np.full((n, n), float(missing_probability), dtype=np.float32)
    np.fill_diagonal(matrix, 1.0)
    for (i, j), value in zip(pairs, probabilities):
        matrix[i, j] = matrix[j, i] = float(value)
    return matrix
