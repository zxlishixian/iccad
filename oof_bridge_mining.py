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
    quality: float = 1.0  # 0~1 quality score for weighted loss


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


def bridge_quality_score(
    a: dict, b: dict,
    oof_probability: float,
    fragment_size_i: int = 1,
    fragment_size_j: int = 1,
    oof_ba: float | None = None,
) -> float:
    """Score bridge edge quality 0~1 for weighted loss.

    High quality = low OOF prob, shared structural signals, low conflict, large fragments.

    Components:
      - oof_confidence: 1 - oof_probability (lower prob → higher quality)
      - signal_agreement: same primary_type, mismatch_type, op_pair
      - no_conflict: 0 if strong conflict, 1 if none
      - fragment_penalty: penalize very small fragments (noise)
      - oof_reliability: down-weight if OOF BA is low
    """
    score = 1.0

    # 1. OOF confidence (lower prob → higher quality bridge)
    oof_conf = 1.0 - float(oof_probability)
    score *= 0.5 + 0.5 * oof_conf  # range [0.5, 1.0] on oof confidence

    # 2. Signal agreement on key fields
    agree = 0
    total = 0
    for key in ("primary_type", "mismatch_type"):
        av, bv = _value(a, key), _value(b, key)
        if av and bv:
            total += 1
            if av == bv:
                agree += 1
    if total > 0:
        agree_ratio = agree / total
        score *= 0.3 + 0.7 * agree_ratio  # range [0.3, 1.0]

    # 3. Same primary_signature is a strong positive signal
    ps_a = _value(a, "primary_signature")
    ps_b = _value(b, "primary_signature")
    if ps_a and ps_b and ps_a == ps_b:
        score *= 1.0  # no penalty
    elif ps_a and ps_b:
        score *= 0.7  # different primary signatures → lower quality

    # 4. Conflict penalty: strong conflict → low quality
    if strong_structured_conflict(a, b):
        score *= 0.3

    # 5. Fragment size penalty: very small fragments are likely noise
    min_frag_size = min(fragment_size_i, fragment_size_j)
    if min_frag_size <= 1:
        score *= 0.5
    elif min_frag_size <= 2:
        score *= 0.7
    elif min_frag_size <= 3:
        score *= 0.85

    # 6. OOF reliability: if OOF model is poor, all bridges are less reliable
    if oof_ba is not None and oof_ba < 0.65:
        score *= 0.3 + 0.7 * (oof_ba / 0.65)

    return max(0.05, min(1.0, score))


def _count_conflicts(a: dict, b: dict) -> int:
    c = 0
    for key in ("primary_type", "mismatch_type", "op_pair"):
        av, bv = _value(a, key), _value(b, key)
        if av and bv and av != bv:
            c += 1
    return c


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
    quality_weighted: bool = False,
    quality_min: float = 0.2,
    top_quality_ratio: float | None = None,
    max_edges_total: int | None = None,
    hardest_fragments_only: bool = False,
    oof_ba: float | None = None,
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

    all_candidates: list[BridgeEdge] = []
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

        fragment_items = sorted(fragments.items())

        # ── Hardest fragments only: compute fragment distances ──
        fragment_scores: list[tuple[int, int, float]] = []  # (frag_i, frag_j, distance_score)
        if hardest_fragments_only and len(fragment_items) > 1:
            # Compute centroid of each fragment using OOF probabilities
            for left_pos, (fi, left) in enumerate(fragment_items):
                for fj, right in fragment_items[left_pos + 1:]:
                    # Mean OOF prob between fragments = distance
                    cross_probs = []
                    for i in left:
                        for j in right:
                            cross_probs.append(float(oof_prob_matrix[i, j]))
                    mean_cross_prob = float(np.mean(cross_probs)) if cross_probs else 1.0
                    fragment_scores.append((fi, fj, mean_cross_prob))
            # Only keep fragments with the LOWEST cross-prob (hardest/most separated)
            if len(fragment_scores) > 1:
                fragment_scores.sort(key=lambda x: x[2])
                # Keep top half (hardest fragments to merge)
                keep_fragments = set()
                for fi, fj, _ in fragment_scores[:max(1, len(fragment_scores) // 2)]:
                    keep_fragments.add((fi, fj))
                    keep_fragments.add((fj, fi))
            else:
                keep_fragments = {(fs[0], fs[1]) for fs in fragment_scores}
                keep_fragments.update({(fs[1], fs[0]) for fs in fragment_scores})

        bug_candidates: list[BridgeEdge] = []
        for left_pos, (fragment_i, left) in enumerate(fragment_items):
            for fragment_j, right in fragment_items[left_pos + 1:]:
                if hardest_fragments_only and (fragment_i, fragment_j) not in keep_fragments:
                    continue
                candidates: list[BridgeEdge] = []
                for i in left:
                    for j in right:
                        probability = float(oof_prob_matrix[i, j])
                        if probability >= cutoff:
                            continue
                        if conflict_filter and strong_structured_conflict(pair_infos[i], pair_infos[j]):
                            continue
                        quality = bridge_quality_score(
                            pair_infos[i], pair_infos[j], probability,
                            fragment_size_i=len(left), fragment_size_j=len(right),
                            oof_ba=oof_ba,
                        )
                        if quality < quality_min:
                            continue
                        candidates.append(BridgeEdge(
                            i=min(i, j), j=max(i, j),
                            weight=quality if quality_weighted else 1.0,
                            bug_id=bug_id,
                            fragment_i=fragment_i, fragment_j=fragment_j,
                            oof_probability=probability,
                            quality=quality,
                        ))
                if quality_weighted:
                    candidates.sort(key=lambda edge: edge.quality, reverse=True)
                else:
                    candidates.sort(key=lambda edge: edge.oof_probability)
                bug_candidates.extend(candidates[:max_edges_per_fragment_pair])

        if quality_weighted:
            bug_candidates.sort(key=lambda edge: edge.quality, reverse=True)
        else:
            bug_candidates.sort(key=lambda edge: edge.oof_probability)
        all_candidates.extend(bug_candidates[:max_edges_per_bug])

    # ── Top-quality ratio filter ──
    if top_quality_ratio is not None and all_candidates:
        all_candidates.sort(key=lambda e: e.quality, reverse=True)
        keep = max(1, int(round(len(all_candidates) * top_quality_ratio)))
        all_candidates = all_candidates[:keep]

    # ── Total budget cap ──
    if max_edges_total is not None and len(all_candidates) > max_edges_total:
        if quality_weighted:
            all_candidates.sort(key=lambda e: e.quality, reverse=True)
        else:
            all_candidates.sort(key=lambda e: e.oof_probability)
        all_candidates = all_candidates[:max_edges_total]

    # ── Deduplicate (keep highest-quality duplicate) ──
    seen: dict[tuple[int, int], BridgeEdge] = {}
    for e in all_candidates:
        key = (e.i, e.j)
        if key not in seen or e.quality > seen[key].quality:
            seen[key] = e
    return sorted(seen.values(), key=lambda e: (e.bug_id, e.oof_probability))


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
