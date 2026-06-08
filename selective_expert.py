#!/usr/bin/env python3
"""Difficulty scoring, selective expert masks, and learned probability gating."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class DifficultyResult:
    scores: np.ndarray
    entropy: np.ndarray
    disagreement: np.ndarray
    margin: np.ndarray
    instability: np.ndarray
    selected: np.ndarray


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / math.log(2.0)


def _rank01(values: np.ndarray, reverse: bool = False) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float32)
    order = np.argsort(-values if reverse else values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks


def _cluster_affinity_margin(prob: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    n = len(labels)
    out = np.ones(n, dtype=np.float32)
    clusters = sorted(set(int(x) for x in labels))
    for idx in range(n):
        scores = []
        for cluster in clusters:
            members = [j for j, value in enumerate(labels) if int(value) == cluster and j != idx]
            scores.append(float(np.mean(prob[idx, members])) if members else 0.0)
        scores.sort(reverse=True)
        out[idx] = scores[0] - scores[1] if len(scores) >= 2 else 1.0
    return out


def _assignment_instability(
    prob: np.ndarray,
    labels: Sequence[int],
    k: int,
    random_state: int,
    repeats: int = 3,
) -> np.ndarray:
    from pairwise_llm_features import cluster_from_probability

    base_same = np.equal.outer(labels, labels)
    rng = np.random.default_rng(random_state)
    accum = np.zeros(len(labels), dtype=np.float32)
    for _ in range(repeats):
        noise = rng.normal(0.0, 0.025, size=prob.shape).astype(np.float32)
        noise = (noise + noise.T) * 0.5
        perturbed = np.clip(prob + noise, 0.0, 1.0)
        np.fill_diagonal(perturbed, 1.0)
        trial = cluster_from_probability(perturbed, k)
        trial_same = np.equal.outer(trial, trial)
        accum += np.mean(base_same != trial_same, axis=1).astype(np.float32)
    return accum / max(1, repeats)


def compute_case_difficulty(
    prob: np.ndarray,
    labels: Sequence[int],
    k: int,
    view_probabilities: Sequence[np.ndarray] | None = None,
    budget: float = 0.15,
    min_cases: int = 2,
    max_cases: int = 20,
    random_state: int = 0,
) -> DifficultyResult:
    n = prob.shape[0]
    offdiag = ~np.eye(n, dtype=bool)
    entropy_matrix = _entropy(prob)
    entropy = np.asarray([
        float(np.mean(entropy_matrix[i, offdiag[i]])) if n > 1 else 0.0
        for i in range(n)
    ], dtype=np.float32)
    disagreement = np.zeros(n, dtype=np.float32)
    if view_probabilities and len(view_probabilities) >= 2:
        stack = np.stack(view_probabilities, axis=0)
        disagreement = np.asarray([
            float(np.mean(np.std(stack[:, i, offdiag[i]], axis=0))) if n > 1 else 0.0
            for i in range(n)
        ], dtype=np.float32)
    affinity_margin = _cluster_affinity_margin(prob, labels)
    instability = _assignment_instability(prob, labels, k, random_state)
    scores = (
        0.35 * _rank01(entropy)
        + 0.25 * _rank01(disagreement)
        + 0.20 * _rank01(affinity_margin, reverse=True)
        + 0.20 * _rank01(instability)
    ).astype(np.float32)
    count = min(max_cases, max(min_cases, int(math.ceil(max(0.0, budget) * n))))
    count = min(n, count)
    selected = np.zeros(n, dtype=bool)
    if count:
        selected[np.argsort(-scores, kind="stable")[:count]] = True
    return DifficultyResult(scores, entropy, disagreement, affinity_margin, instability, selected)


def selective_pair_mask(
    pairs: Sequence[tuple[int, int]],
    selected_cases: Sequence[bool],
) -> np.ndarray:
    selected = np.asarray(selected_cases, dtype=bool)
    return np.asarray([selected[i] or selected[j] for i, j in pairs], dtype=bool)


def build_gate_feature_matrix(
    pairs: Sequence[tuple[int, int]],
    prob_base: np.ndarray,
    prob_expert: np.ndarray,
    difficulty: DifficultyResult,
    conflict_matrix: np.ndarray,
    expert_case_mask: Sequence[bool],
) -> np.ndarray:
    mask = np.asarray(expert_case_mask, dtype=bool)
    rows = np.empty((len(pairs), 14), dtype=np.float32)
    for row, (i, j) in enumerate(pairs):
        pb = float(prob_base[i, j])
        pe = float(prob_expert[i, j])
        rows[row] = (
            pb,
            pe,
            abs(pe - pb),
            float(_entropy(np.asarray([pb]))[0]),
            difficulty.scores[i],
            difficulty.scores[j],
            min(difficulty.scores[i], difficulty.scores[j]),
            max(difficulty.scores[i], difficulty.scores[j]),
            difficulty.entropy[i],
            difficulty.entropy[j],
            difficulty.instability[i],
            difficulty.instability[j],
            float(conflict_matrix[i, j]),
            float(mask[i] or mask[j]),
        )
    return rows


def train_gate(
    X: np.ndarray,
    y: np.ndarray,
    prob_base: np.ndarray,
    prob_expert: np.ndarray,
    random_state: int,
) -> dict:
    y = np.asarray(y, dtype=np.float32)
    pb = np.clip(np.asarray(prob_base, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    pe = np.clip(np.asarray(prob_expert, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    base_loss = -(y * np.log(pb) + (1.0 - y) * np.log(1.0 - pb))
    expert_loss = -(y * np.log(pe) + (1.0 - y) * np.log(1.0 - pe))
    target = (expert_loss + 1e-4 < base_loss).astype(np.int64)
    if len(set(target.tolist())) < 2 or len(target) < 16:
        return {"model_type": "constant", "value": float(np.mean(target)) if len(target) else 0.0}
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=3,
        learning_rate=0.05,
        l2_regularization=0.05,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(X, target)
    return {"model_type": "gbdt", "model": model}


def predict_gate(model_pkg: dict, X: np.ndarray, availability: np.ndarray) -> np.ndarray:
    if model_pkg.get("model_type") == "constant":
        gate = np.full(len(X), float(model_pkg.get("value", 0.0)), dtype=np.float32)
    else:
        gate = model_pkg["model"].predict_proba(X)[:, 1].astype(np.float32)
    return gate * np.asarray(availability, dtype=np.float32)


def fuse_probability_matrix(
    prob_base: np.ndarray,
    prob_expert: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    gate_values: Sequence[float],
) -> np.ndarray:
    out = prob_base.astype(np.float32, copy=True)
    for (i, j), gate in zip(pairs, gate_values):
        value = (1.0 - float(gate)) * float(prob_base[i, j]) + float(gate) * float(prob_expert[i, j])
        out[i, j] = out[j, i] = value
    np.fill_diagonal(out, 1.0)
    return out
