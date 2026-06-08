from __future__ import annotations

import numpy as np

from oof_bridge_mining import fragmentation_rows, mine_oof_bridge_edges, strong_structured_conflict


def test_mines_only_cross_fragment_low_probability_edges() -> None:
    cases = ["a", "b", "c", "d"]
    gold = ["bug1", "bug1", "bug1", "bug2"]
    pred = [0, 0, 1, 2]
    prob = np.eye(4, dtype=np.float32)
    prob[0, 1] = prob[1, 0] = 0.8
    prob[0, 2] = prob[2, 0] = 0.1
    prob[1, 2] = prob[2, 1] = 0.4
    infos = [{} for _ in cases]
    edges = mine_oof_bridge_edges(cases, gold, prob, pred, infos, bridge_threshold=0.35)
    assert [(edge.i, edge.j) for edge in edges] == [(0, 2)]


def test_conflict_filter_rejects_three_way_conflict() -> None:
    a = {"primary_type": "fatal", "mismatch_type": "pc", "op_pair": "add/sub"}
    b = {"primary_type": "timeout", "mismatch_type": "reg", "op_pair": "mul/div"}
    assert strong_structured_conflict(a, b)
    prob = np.eye(2, dtype=np.float32); prob[0, 1] = prob[1, 0] = 0.1
    assert mine_oof_bridge_edges(["a", "b"], ["bug", "bug"], prob, [0, 1], [a, b]) == []


def test_fragmentation_metrics() -> None:
    rows = fragmentation_rows(["a", "b", "c"], ["bug", "bug", "bug"], [0, 0, 1])
    assert rows[0]["num_pred_fragments"] == 2
    assert rows[0]["largest_fragment_ratio"] == 2 / 3
    assert rows[0]["intra_bug_TPR"] == 1 / 3
    assert rows[0]["fn_pairs"] == 2
