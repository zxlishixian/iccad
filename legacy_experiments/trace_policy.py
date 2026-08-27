#!/usr/bin/env python3
"""Conservative trace veto/boost policies for experimental postprocess."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

import pairwise_llm_features as plf
import trace_features as tf


@dataclass
class TracePolicyParams:
    trace_policy: str = "none"
    veto_base_min: float = 0.65
    veto_trace_max: float = 0.10
    veto_cap: float = 0.35
    boost_base_low: float = 0.30
    boost_base_high: float = 0.65
    boost_trace_min: float = 0.75
    boost_floor: float = 0.65


@dataclass
class TracePolicyStats:
    pairs_vetoed: int = 0
    pairs_boosted: int = 0
    trace_missing_pairs: int = 0
    candidate_pairs: int = 0


def _same_nonempty(a: str, b: str) -> bool:
    return bool(a and b and a == b)


def _conflict(a: str, b: str) -> bool:
    return bool(a and b and a != b)


def _info(case_features: Sequence[plf.LLMCaseFeature], idx: int, key: str) -> str:
    value = (case_features[idx].info or {}).get(key, "")
    return str(value) if value is not None else ""


def strong_structured_agreement(case_features: Sequence[plf.LLMCaseFeature], i: int, j: int) -> bool:
    primary_sig = _same_nonempty(_info(case_features, i, "primary_signature"), _info(case_features, j, "primary_signature"))
    op_pair = _same_nonempty(_info(case_features, i, "op_pair"), _info(case_features, j, "op_pair"))
    mismatch = _same_nonempty(_info(case_features, i, "mismatch_type"), _info(case_features, j, "mismatch_type"))
    primary_type = _same_nonempty(_info(case_features, i, "primary_type"), _info(case_features, j, "primary_type"))
    return primary_sig or op_pair or (mismatch and primary_type)


def structured_conflict(case_features: Sequence[plf.LLMCaseFeature], i: int, j: int) -> bool:
    return (
        _conflict(_info(case_features, i, "primary_type"), _info(case_features, j, "primary_type"))
        or _conflict(_info(case_features, i, "mismatch_type"), _info(case_features, j, "mismatch_type"))
        or _conflict(_info(case_features, i, "op_pair"), _info(case_features, j, "op_pair"))
    )


def trace_agreement(a: tf.TraceCaseFeature, b: tf.TraceCaseFeature) -> float:
    vec = tf.build_trace_pair_feature_vector(a, b)
    opcode_jaccard = float(vec[0])
    pc_region_jaccard = float(vec[4])
    tail_lcs_ratio = float(vec[6])
    tail_ngram2_jaccard = float(vec[7])
    return (
        0.30 * opcode_jaccard
        + 0.25 * tail_ngram2_jaccard
        + 0.25 * tail_lcs_ratio
        + 0.20 * pc_region_jaccard
    )


def apply_trace_policy(
    prob_base: np.ndarray,
    trace_features: Sequence[tf.TraceCaseFeature],
    case_features: Sequence[plf.LLMCaseFeature],
    params: TracePolicyParams,
) -> tuple[np.ndarray, TracePolicyStats]:
    policy = params.trace_policy
    out = prob_base.astype(np.float32, copy=True)
    stats = TracePolicyStats()
    if policy == "none":
        return out, stats
    if policy not in {"veto", "boost", "veto_boost"}:
        raise ValueError(f"unknown trace_policy: {policy}")
    n = out.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if trace_features[i].missing or trace_features[j].missing:
                stats.trace_missing_pairs += 1
                continue
            p = float(out[i, j])
            agree = trace_agreement(trace_features[i], trace_features[j])
            stats.candidate_pairs += 1
            if policy in {"veto", "veto_boost"}:
                if (
                    p >= params.veto_base_min
                    and agree <= params.veto_trace_max
                    and not strong_structured_agreement(case_features, i, j)
                ):
                    new_p = min(p, float(params.veto_cap))
                    if new_p < p:
                        out[i, j] = out[j, i] = new_p
                        stats.pairs_vetoed += 1
                    p = new_p
            if policy in {"boost", "veto_boost"}:
                if (
                    params.boost_base_low <= p <= params.boost_base_high
                    and agree >= params.boost_trace_min
                    and not structured_conflict(case_features, i, j)
                ):
                    new_p = max(p, float(params.boost_floor))
                    if new_p > p:
                        out[i, j] = out[j, i] = new_p
                        stats.pairs_boosted += 1
    np.fill_diagonal(out, 1.0)
    return out, stats
