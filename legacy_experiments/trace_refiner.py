#!/usr/bin/env python3
"""Selective trace-assisted pair probability refinement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

import pairwise_llm_features as plf
import trace_features as tf


@dataclass
class RefineStats:
    uncertain_pairs: int = 0
    refined_pairs: int = 0
    trace_missing_pairs: int = 0


def uncertain_pairs_from_probability(prob: np.ndarray, lower: float, upper: float) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    n = int(prob.shape[0])
    lo = float(lower)
    hi = float(upper)
    for i in range(n):
        for j in range(i + 1, n):
            p = float(prob[i, j])
            if lo <= p <= hi:
                pairs.append((i, j))
    return pairs


def pair_labels_from_gold(gold: Sequence[str], pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    return np.asarray([1.0 if gold[i] == gold[j] else 0.0 for i, j in pairs], dtype=np.float32)


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-5), 1.0 - 1e-5)
    return math.log(p / (1.0 - p))


def _base_pair_matrix(prob: np.ndarray, pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    X = np.empty((len(pairs), 3), dtype=np.float32)
    for row, (i, j) in enumerate(pairs):
        p = float(prob[i, j])
        X[row] = (p, _logit(p), abs(p - 0.5))
    return X


def _summary_pair_matrix(
    case_features: Sequence[plf.LLMCaseFeature] | None,
    pairs: Sequence[tuple[int, int]],
) -> np.ndarray:
    if case_features is None:
        return np.zeros((len(pairs), 0), dtype=np.float32)
    return plf.build_rich_pair_feature_matrix(list(case_features), list(pairs), feature_mode="summary21")


def build_refiner_feature_matrix(
    prob: np.ndarray,
    trace_features: Sequence[tf.TraceCaseFeature],
    pairs: Sequence[tuple[int, int]],
    case_features: Sequence[plf.LLMCaseFeature] | None = None,
    include_summary: bool = True,
) -> np.ndarray:
    blocks = [
        _base_pair_matrix(prob, pairs),
        tf.build_trace_pair_feature_matrix(trace_features, pairs),
    ]
    if include_summary:
        blocks.append(_summary_pair_matrix(case_features, pairs))
    return np.hstack(blocks).astype(np.float32, copy=False)


def count_trace_missing_pairs(trace_features: Sequence[tf.TraceCaseFeature], pairs: Sequence[tuple[int, int]]) -> int:
    return sum(1 for i, j in pairs if trace_features[i].missing or trace_features[j].missing)


def train_trace_refiner(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "gbdt",
    random_state: int = 0,
) -> dict:
    y = y.astype(np.float32, copy=False)
    classes = sorted(set(float(v) for v in y.tolist()))
    if X.shape[0] == 0 or len(classes) < 2:
        value = float(y.mean()) if len(y) else 0.5
        return {"model_type": "constant", "constant_prob": value, "include_summary": True}
    if model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        )
        model.fit(Xs, y)
        return {"model_type": "logistic", "model": model, "scaler": scaler, "include_summary": True}
    if model_type == "gbdt":
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(
            max_iter=160,
            max_depth=4,
            learning_rate=0.05,
            l2_regularization=0.02,
            early_stopping=True,
            random_state=random_state,
            class_weight="balanced",
        )
        model.fit(X, y)
        return {"model_type": "gbdt", "model": model, "scaler": None, "include_summary": True}
    raise ValueError(f"unknown refiner model_type: {model_type}")


def predict_refiner_proba(model_pkg: dict, X: np.ndarray) -> np.ndarray:
    if X.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    model_type = str(model_pkg.get("model_type", ""))
    if model_type == "constant":
        return np.full(X.shape[0], float(model_pkg.get("constant_prob", 0.5)), dtype=np.float32)
    scaler = model_pkg.get("scaler")
    if scaler is not None:
        X = scaler.transform(X)
    model = model_pkg["model"]
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1].astype(np.float32)
    return np.clip(model.predict(X).astype(np.float32), 1e-5, 1.0 - 1e-5)


def refine_probability_matrix(
    prob_base: np.ndarray,
    trace_case_features: Sequence[tf.TraceCaseFeature],
    refiner_pkg: dict,
    lower: float,
    upper: float,
    case_features: Sequence[plf.LLMCaseFeature] | None = None,
    skip_missing_trace: bool = True,
) -> tuple[np.ndarray, RefineStats]:
    pairs = uncertain_pairs_from_probability(prob_base, lower, upper)
    stats = RefineStats(uncertain_pairs=len(pairs))
    stats.trace_missing_pairs = count_trace_missing_pairs(trace_case_features, pairs)
    if skip_missing_trace:
        pairs = [(i, j) for i, j in pairs if not (trace_case_features[i].missing or trace_case_features[j].missing)]
    stats.refined_pairs = len(pairs)
    out = prob_base.astype(np.float32, copy=True)
    if not pairs:
        return out, stats
    X = build_refiner_feature_matrix(
        out,
        trace_case_features,
        pairs,
        case_features=case_features,
        include_summary=bool(refiner_pkg.get("include_summary", True)),
    )
    pred = predict_refiner_proba(refiner_pkg, X)
    for (i, j), p in zip(pairs, pred):
        out[i, j] = float(p)
        out[j, i] = float(p)
    np.fill_diagonal(out, 1.0)
    return out, stats
