from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def choose_cluster_anchors(
    labels: np.ndarray,
    similarity: np.ndarray,
    selected: np.ndarray,
    anchors_per_cluster: int,
) -> dict[int, list[int]]:
    anchors: dict[int, list[int]] = {}
    for label in sorted(set(int(value) for value in labels)):
        members = np.flatnonzero(labels == label)
        candidates = members[~selected[members]]
        if len(candidates) < anchors_per_cluster:
            candidates = members
        centrality = np.asarray([
            float(np.mean(similarity[idx, members[members != idx]]))
            if len(members) > 1 else 1.0
            for idx in candidates
        ])
        order = np.argsort(-centrality, kind="mergesort")
        anchors[label] = [int(candidates[idx]) for idx in order[:anchors_per_cluster]]
    return anchors


def sparse_refine_labels(
    base_labels: np.ndarray,
    deterministic_similarity: np.ndarray,
    expert_probability: np.ndarray,
    selected: np.ndarray,
    anchors_per_cluster: int,
    expert_weight: float,
    min_probability: float,
    margin: float,
) -> tuple[np.ndarray, dict[str, int | float]]:
    labels = np.asarray(base_labels, dtype=np.int32).copy()
    anchors = choose_cluster_anchors(labels, deterministic_similarity, selected, anchors_per_cluster)
    counts = {label: int(np.sum(labels == label)) for label in anchors}
    moved = 0
    evaluated_edges: set[tuple[int, int]] = set()
    for idx in np.flatnonzero(selected):
        own_label = int(labels[idx])
        cluster_scores: dict[int, float] = {}
        for label, cluster_anchors in anchors.items():
            values = []
            for anchor in cluster_anchors:
                if anchor == idx:
                    continue
                edge = (min(int(idx), anchor), max(int(idx), anchor))
                evaluated_edges.add(edge)
                score = expert_weight * float(expert_probability[idx, anchor]) + (1.0 - expert_weight) * float(deterministic_similarity[idx, anchor])
                values.append(score)
            if values:
                cluster_scores[label] = float(np.mean(values))
        if own_label not in cluster_scores or not cluster_scores:
            continue
        best_label, best_score = max(cluster_scores.items(), key=lambda item: item[1])
        own_score = cluster_scores[own_label]
        if best_label != own_label and counts[own_label] > 1 and best_score >= min_probability and best_score - own_score >= margin:
            labels[idx] = best_label
            counts[own_label] -= 1
            counts[best_label] += 1
            moved += 1
    return labels, {
        "moved_cases": moved,
        "expert_edges": len(evaluated_edges),
        "anchor_cases": len({idx for values in anchors.values() for idx in values}),
    }


def normalized_det_vectors(features: Sequence[Any]) -> np.ndarray:
    matrix = np.vstack([np.asarray(feature.det_vec, dtype=np.float32) for feature in features])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, np.float32(1e-12))


def _rank01(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if len(values) <= 1:
        return np.zeros(len(values), dtype=np.float32)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(values))
    return ranks.astype(np.float32)


def centroid_sparse_plan(
    base_labels: np.ndarray,
    det_vectors: np.ndarray,
    fraction: float,
    max_selected: int,
    anchors_per_cluster: int,
) -> tuple[np.ndarray, dict[int, list[int]], dict[str, float]]:
    labels = np.asarray(base_labels, dtype=np.int32)
    unique_labels = np.asarray(sorted(set(int(value) for value in labels)), dtype=np.int32)
    centroids = []
    for label in unique_labels:
        centroid = np.mean(det_vectors[labels == label], axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        centroids.append(centroid)
    centroid_matrix = np.asarray(centroids, dtype=np.float32)
    similarities = det_vectors @ centroid_matrix.T
    label_to_col = {int(label): idx for idx, label in enumerate(unique_labels)}
    own_cols = np.asarray([label_to_col[int(label)] for label in labels], dtype=np.int64)
    own_similarity = similarities[np.arange(len(labels)), own_cols]
    other_similarity = similarities.copy()
    other_similarity[np.arange(len(labels)), own_cols] = -np.inf
    nearest_other = np.max(other_similarity, axis=1) if len(unique_labels) > 1 else np.zeros(len(labels))
    margin_difficulty = nearest_other - own_similarity
    difficulty = 0.70 * _rank01(margin_difficulty) + 0.30 * _rank01(-own_similarity)
    count = max(2, int(np.ceil(fraction * len(labels)))) if fraction > 0 else 0
    count = min(len(labels), max_selected if max_selected > 0 else len(labels), count)
    selected = np.zeros(len(labels), dtype=bool)
    selected[np.argsort(-difficulty, kind="mergesort")[:count]] = True
    anchors: dict[int, list[int]] = {}
    for label in unique_labels:
        members = np.flatnonzero(labels == label)
        candidates = members[~selected[members]]
        if len(candidates) < anchors_per_cluster:
            candidates = members
        column = label_to_col[int(label)]
        order = np.argsort(-similarities[candidates, column], kind="mergesort")
        anchors[int(label)] = [int(candidates[idx]) for idx in order[:anchors_per_cluster]]
    return selected, anchors, {
        "difficulty_min": float(np.min(difficulty)),
        "difficulty_mean": float(np.mean(difficulty)),
        "difficulty_max": float(np.max(difficulty)),
    }


def sparse_refine_from_edges(
    base_labels: np.ndarray,
    det_vectors: np.ndarray,
    expert_edges: dict[tuple[int, int], float],
    selected: np.ndarray,
    anchors: dict[int, list[int]],
    expert_weight: float,
    min_probability: float,
    margin: float,
    structured_infos: Sequence[dict] | None = None,
    min_support: int = 1,
    support_probability: float = 0.0,
    max_conflict_ratio: float = 1.0,
    min_det_margin: float = -1.0,
    require_structured_agreement: bool = False,
) -> tuple[np.ndarray, dict[str, int | float]]:
    labels = np.asarray(base_labels, dtype=np.int32).copy()
    counts = {label: int(np.sum(labels == label)) for label in anchors}
    moved = 0
    evaluated = 0
    rejected_support = 0
    rejected_conflict = 0
    rejected_det_margin = 0
    rejected_agreement = 0

    def structured_relation(left: int, right: int) -> tuple[int, int, bool]:
        if structured_infos is None:
            return 0, 0, False
        a = structured_infos[left]
        b = structured_infos[right]
        comparable = 0
        conflicts = 0
        agreement = False
        for field in ("primary_type", "mismatch_type", "op_pair", "register_name"):
            av = str(a.get(field, "") or "")
            bv = str(b.get(field, "") or "")
            if not av or not bv:
                continue
            comparable += 1
            if av == bv:
                agreement = True
            else:
                conflicts += 1
        for field in ("primary_signature", "fatal_file"):
            av = str(a.get(field, "") or "")
            bv = str(b.get(field, "") or "")
            if av and bv and av == bv:
                agreement = True
        return conflicts, comparable, agreement

    for idx in np.flatnonzero(selected):
        own_label = int(labels[idx])
        cluster_scores: dict[int, float] = {}
        cluster_support: dict[int, int] = {}
        cluster_det_scores: dict[int, float] = {}
        cluster_conflict: dict[int, float] = {}
        cluster_agreement: dict[int, bool] = {}
        for label, cluster_anchors in anchors.items():
            values = []
            det_values = []
            support = 0
            conflicts = 0
            comparable = 0
            agreement = False
            for anchor in cluster_anchors:
                if anchor == idx:
                    continue
                edge = (min(int(idx), anchor), max(int(idx), anchor))
                if edge not in expert_edges:
                    continue
                evaluated += 1
                det_probability = float(np.clip(
                    (np.dot(det_vectors[idx], det_vectors[anchor]) + 1.0) * 0.5,
                    0.0, 1.0,
                ))
                values.append(
                    expert_weight * expert_edges[edge]
                    + (1.0 - expert_weight) * det_probability
                )
                det_values.append(det_probability)
                support += int(expert_edges[edge] >= support_probability)
                edge_conflicts, edge_comparable, edge_agreement = structured_relation(
                    int(idx), anchor
                )
                conflicts += edge_conflicts
                comparable += edge_comparable
                agreement = agreement or edge_agreement
            if values:
                cluster_scores[label] = float(np.mean(values))
                cluster_support[label] = support
                cluster_det_scores[label] = float(np.mean(det_values))
                cluster_conflict[label] = conflicts / comparable if comparable else 0.0
                cluster_agreement[label] = agreement
        if own_label not in cluster_scores:
            continue
        best_label, best_score = max(cluster_scores.items(), key=lambda item: item[1])
        own_score = cluster_scores[own_label]
        if (
            best_label == own_label
            or counts[own_label] <= 1
            or best_score < min_probability
            or best_score - own_score < margin
        ):
            continue
        if cluster_support.get(best_label, 0) < min_support:
            rejected_support += 1
            continue
        if cluster_conflict.get(best_label, 0.0) > max_conflict_ratio:
            rejected_conflict += 1
            continue
        det_margin = (
            cluster_det_scores.get(best_label, 0.0)
            - cluster_det_scores.get(own_label, 0.0)
        )
        if det_margin < min_det_margin:
            rejected_det_margin += 1
            continue
        if require_structured_agreement and not cluster_agreement.get(best_label, False):
            rejected_agreement += 1
            continue
        labels[idx] = best_label
        counts[own_label] -= 1
        counts[best_label] += 1
        moved += 1
    return labels, {
        "moved_cases": moved,
        "expert_edges": len(expert_edges),
        "evaluated_anchor_scores": evaluated,
        "anchor_cases": len({idx for values in anchors.values() for idx in values}),
        "rejected_support": rejected_support,
        "rejected_conflict": rejected_conflict,
        "rejected_det_margin": rejected_det_margin,
        "rejected_agreement": rejected_agreement,
    }
