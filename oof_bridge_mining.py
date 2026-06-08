#!/usr/bin/env python3
"""Experimental OOF bridge-edge mining for fragmented same-bug clusters."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BridgeEdge:
    i: int
    j: int
    weight: float
    bug_id: str
    fragment_i: int
    fragment_j: int
    oof_probability: float


def _value(info: dict, key: str) -> str:
    return str(info.get(key, "") or "").strip().lower()


def strong_structured_conflict(a: dict, b: dict) -> bool:
    """Reject only pairs with simultaneous disagreement in three strong fields."""
    conflicts = []
    for key in ("primary_type", "mismatch_type", "op_pair"):
        av, bv = _value(a, key), _value(b, key)
        conflicts.append(bool(av and bv and av != bv))
    fatal_a = _value(a, "fatal_file") or _value(a, "error_source_file")
    fatal_b = _value(b, "fatal_file") or _value(b, "error_source_file")
    fatal_conflict = bool(fatal_a and fatal_b and fatal_a != fatal_b)
    return all(conflicts) or (sum(conflicts) >= 2 and fatal_conflict)


def mine_oof_bridge_edges(
    cases: Sequence[str],
    gold_bug_ids: Sequence[str],
    oof_prob_matrix: np.ndarray,
    oof_pred_labels: Sequence[int],
    pair_infos: Sequence[dict],
    bridge_select: str = "abs_threshold",
    bridge_threshold: float = 0.35,
    bridge_quantile: float = 0.2,
    max_edges_per_bug: int = 200,
    max_edges_per_fragment_pair: int = 20,
    conflict_filter: bool = True,
) -> list[BridgeEdge]:
    if bridge_select not in {"abs_threshold", "bug_quantile"}:
        raise ValueError(f"unknown bridge_select: {bridge_select}")
    n = len(gold_bug_ids)
    if not (len(cases) == n == len(oof_pred_labels) == len(pair_infos)):
        raise ValueError("case/label/pred/info length mismatch")
    if oof_prob_matrix.shape != (n, n):
        raise ValueError(f"probability shape mismatch: {oof_prob_matrix.shape} vs {(n, n)}")

    by_bug: dict[str, list[int]] = defaultdict(list)
    for idx, bug in enumerate(gold_bug_ids):
        by_bug[str(bug)].append(idx)

    selected: list[BridgeEdge] = []
    for bug_id, members in sorted(by_bug.items()):
        fragments: dict[int, list[int]] = defaultdict(list)
        for idx in members:
            fragments[int(oof_pred_labels[idx])].append(idx)
        if len(fragments) <= 1:
            continue

        same_bug_probs = [
            float(oof_prob_matrix[i, j])
            for pos, i in enumerate(members)
            for j in members[pos + 1:]
        ]
        quantile_cutoff = float(np.quantile(same_bug_probs, bridge_quantile)) if same_bug_probs else bridge_threshold
        cutoff = float(bridge_threshold) if bridge_select == "abs_threshold" else quantile_cutoff

        bug_candidates: list[BridgeEdge] = []
        fragment_items = sorted(fragments.items())
        for left_pos, (fragment_i, left) in enumerate(fragment_items):
            for fragment_j, right in fragment_items[left_pos + 1:]:
                candidates: list[BridgeEdge] = []
                for i in left:
                    for j in right:
                        probability = float(oof_prob_matrix[i, j])
                        if probability >= cutoff:
                            continue
                        if conflict_filter and strong_structured_conflict(pair_infos[i], pair_infos[j]):
                            continue
                        candidates.append(BridgeEdge(
                            i=min(i, j), j=max(i, j), weight=1.0, bug_id=bug_id,
                            fragment_i=fragment_i, fragment_j=fragment_j,
                            oof_probability=probability,
                        ))
                candidates.sort(key=lambda edge: edge.oof_probability)
                bug_candidates.extend(candidates[:max_edges_per_fragment_pair])
        bug_candidates.sort(key=lambda edge: edge.oof_probability)
        selected.extend(bug_candidates[:max_edges_per_bug])
    return selected


def fragmentation_rows(
    cases: Sequence[str],
    gold_bug_ids: Sequence[str],
    predicted_labels: Sequence[int],
) -> list[dict]:
    by_bug: dict[str, list[int]] = defaultdict(list)
    for idx, bug in enumerate(gold_bug_ids):
        by_bug[str(bug)].append(idx)
    rows = []
    for bug_id, members in sorted(by_bug.items()):
        counts: dict[int, int] = defaultdict(int)
        for idx in members:
            counts[int(predicted_labels[idx])] += 1
        total_pairs = len(members) * (len(members) - 1) // 2
        connected_pairs = sum(value * (value - 1) // 2 for value in counts.values())
        rows.append({
            "bug_id": bug_id,
            "num_cases": len(members),
            "num_pred_fragments": len(counts),
            "largest_fragment_ratio": max(counts.values()) / max(1, len(members)),
            "intra_bug_TPR": connected_pairs / total_pairs if total_pairs else 1.0,
            "fn_pairs": total_pairs - connected_pairs,
        })
    return sorted(rows, key=lambda row: (-int(row["num_pred_fragments"]), float(row["largest_fragment_ratio"]), -int(row["num_cases"])))
