#!/usr/bin/env python3
"""Lightweight experimental postprocess for pairwise clustering outputs.

These helpers use only predicted labels, pairwise probabilities, and case info
extracted from sim.log/regr.log. They do not use gold/meta/trace.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class MergeConfig:
    topk: int = 10
    prob_threshold: float = 0.80
    consistency_threshold: float = 0.65
    conflict_max: float = 0.20


@dataclass(frozen=True)
class SplitConfig:
    min_bucket_size: int = 4
    min_group_size: int = 2
    key: str = "auto"


def _buckets(labels: Sequence[int]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        out[int(label)].append(idx)
    return dict(out)


def _info_value(features: Sequence[object], idx: int, key: str) -> str:
    info = getattr(features[idx], "info", {}) or {}
    value = info.get(key, "")
    return str(value) if value is not None else ""


def _same_ratio(features: Sequence[object], left: Sequence[int], right: Sequence[int], key: str) -> float:
    total = 0
    same = 0
    for i in left:
        vi = _info_value(features, i, key)
        if not vi:
            continue
        for j in right:
            vj = _info_value(features, j, key)
            if not vj:
                continue
            total += 1
            if vi == vj:
                same += 1
    return same / total if total else 0.0


def _conflict_ratio(features: Sequence[object], left: Sequence[int], right: Sequence[int]) -> float:
    keys = ("primary_type", "mismatch_type", "op_pair")
    total = 0
    conflicts = 0
    for i in left:
        for j in right:
            for key in keys:
                vi = _info_value(features, i, key)
                vj = _info_value(features, j, key)
                if vi and vj:
                    total += 1
                    if vi != vj:
                        conflicts += 1
    return conflicts / total if total else 0.0


def _topk_mean(prob: np.ndarray, left: Sequence[int], right: Sequence[int], topk: int) -> float:
    vals = prob[np.ix_(list(left), list(right))].ravel()
    if vals.size == 0:
        return 0.0
    k = min(max(1, int(topk)), vals.size)
    if k == vals.size:
        return float(np.mean(vals))
    part = np.partition(vals, vals.size - k)[-k:]
    return float(np.mean(part))


class _DSU:
    def __init__(self, labels: Sequence[int]):
        uniq = sorted(set(int(x) for x in labels))
        self.parent = {x: x for x in uniq}

    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def relabel_dense(labels: Sequence[int]) -> list[int]:
    mapping: dict[int, int] = {}
    out: list[int] = []
    for label in labels:
        label = int(label)
        if label not in mapping:
            mapping[label] = len(mapping)
        out.append(mapping[label])
    return out


def merge_close_buckets(labels: Sequence[int], prob: np.ndarray, features: Sequence[object], config: MergeConfig) -> list[int]:
    labels = [int(x) for x in labels]
    buckets = _buckets(labels)
    dsu = _DSU(labels)
    pairs = []
    keys = ("primary_signature", "mismatch_type", "op_pair")
    bucket_items = list(buckets.items())
    for pos, (la, aidx) in enumerate(bucket_items):
        for lb, bidx in bucket_items[pos + 1:]:
            topk_prob = _topk_mean(prob, aidx, bidx, config.topk)
            if topk_prob < config.prob_threshold:
                continue
            consistency = max(_same_ratio(features, aidx, bidx, key) for key in keys)
            if consistency < config.consistency_threshold:
                continue
            conflict = _conflict_ratio(features, aidx, bidx)
            if conflict > config.conflict_max:
                continue
            pairs.append((topk_prob, consistency, -conflict, la, lb))
    for _score, _cons, _neg_conflict, la, lb in sorted(pairs, reverse=True):
        dsu.union(la, lb)
    return relabel_dense([dsu.find(label) for label in labels])


def _choose_split_key(indices: Sequence[int], features: Sequence[object], config: SplitConfig) -> str | None:
    keys = [config.key] if config.key != "auto" else ["primary_signature", "op_pair", "mismatch_type", "fatal_file"]
    for key in keys:
        groups: dict[str, list[int]] = defaultdict(list)
        for idx in indices:
            value = _info_value(features, idx, key)
            if value:
                groups[value].append(idx)
        valid = [g for g in groups.values() if len(g) >= config.min_group_size]
        if len(valid) >= 2 and sum(len(g) for g in valid) >= max(config.min_bucket_size, len(indices) // 2):
            return key
    return None


def split_mixed_buckets(labels: Sequence[int], features: Sequence[object], config: SplitConfig) -> list[int]:
    labels = [int(x) for x in labels]
    buckets = _buckets(labels)
    new_labels = list(labels)
    next_label = max(labels) + 1 if labels else 0
    for label, indices in buckets.items():
        if len(indices) < config.min_bucket_size:
            continue
        key = _choose_split_key(indices, features, config)
        if not key:
            continue
        groups: dict[str, list[int]] = defaultdict(list)
        unknown: list[int] = []
        for idx in indices:
            value = _info_value(features, idx, key)
            if value:
                groups[value].append(idx)
            else:
                unknown.append(idx)
        valid_values = [value for value, group in groups.items() if len(group) >= config.min_group_size]
        if len(valid_values) < 2:
            continue
        first = True
        for value in sorted(valid_values, key=lambda v: (-len(groups[v]), v)):
            target = label if first else next_label
            if not first:
                next_label += 1
            for idx in groups[value]:
                new_labels[idx] = target
            first = False
        # Leave small/unknown groups in the original label to avoid singleton churn.
    return relabel_dense(new_labels)


def apply_postprocess(labels: Sequence[int], prob: np.ndarray, features: Sequence[object], mode: str, params: dict) -> list[int]:
    if mode == "none":
        return relabel_dense(labels)
    if mode == "merge_close":
        return merge_close_buckets(labels, prob, features, MergeConfig(
            topk=int(params.get("merge_topk", 10)),
            prob_threshold=float(params.get("merge_prob_threshold", 0.80)),
            consistency_threshold=float(params.get("merge_consistency_threshold", 0.65)),
            conflict_max=float(params.get("merge_conflict_max", 0.20)),
        ))
    if mode == "split_mixed":
        return split_mixed_buckets(labels, features, SplitConfig(
            min_bucket_size=int(params.get("split_min_bucket_size", 4)),
            min_group_size=int(params.get("split_min_group_size", 2)),
            key=str(params.get("split_key", "auto")),
        ))
    if mode == "split_then_merge":
        split = split_mixed_buckets(labels, features, SplitConfig(
            min_bucket_size=int(params.get("split_min_bucket_size", 4)),
            min_group_size=int(params.get("split_min_group_size", 2)),
            key=str(params.get("split_key", "auto")),
        ))
        return merge_close_buckets(split, prob, features, MergeConfig(
            topk=int(params.get("merge_topk", 10)),
            prob_threshold=float(params.get("merge_prob_threshold", 0.80)),
            consistency_threshold=float(params.get("merge_consistency_threshold", 0.65)),
            conflict_max=float(params.get("merge_conflict_max", 0.20)),
        ))
    raise ValueError(f"unknown postprocess mode: {mode}")
