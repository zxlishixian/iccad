from __future__ import annotations

import unittest

import numpy as np

from run_sparse_anchor_refinement_experiments import (
    choose_cluster_anchors,
    sparse_refine_labels,
)
from sparse_multiview_inference import build_active_pairs


class SparseMultiviewTests(unittest.TestCase):
    def test_anchor_prefers_unselected_central_member(self):
        labels = np.asarray([0, 0, 0, 1, 1], dtype=np.int32)
        similarity = np.eye(5, dtype=np.float32)
        similarity[0, 1] = similarity[1, 0] = 0.9
        similarity[1, 2] = similarity[2, 1] = 0.9
        similarity[0, 2] = similarity[2, 0] = 0.2
        similarity[3, 4] = similarity[4, 3] = 0.8
        selected = np.asarray([False, True, False, False, False])
        anchors = choose_cluster_anchors(labels, similarity, selected, 1)
        self.assertNotEqual(anchors[0][0], 1)
        self.assertIn(anchors[1][0], {3, 4})

    def test_refinement_moves_only_with_margin(self):
        labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
        det = np.full((4, 4), 0.2, dtype=np.float32)
        np.fill_diagonal(det, 1.0)
        det[0, 1] = det[1, 0] = 0.7
        det[2, 3] = det[3, 2] = 0.7
        expert = det.copy()
        expert[0, 1] = expert[1, 0] = 0.2
        expert[0, 2] = expert[2, 0] = 0.9
        selected = np.asarray([True, False, False, False])
        refined, stats = sparse_refine_labels(
            labels, det, expert, selected, 1, 1.0, 0.55, 0.10
        )
        self.assertEqual(refined.tolist(), [1, 0, 1, 1])
        self.assertEqual(stats["moved_cases"], 1)

    def test_refinement_never_empties_cluster(self):
        labels = np.asarray([0, 1, 1], dtype=np.int32)
        det = np.eye(3, dtype=np.float32)
        expert = np.asarray([
            [1.0, 0.9, 0.9],
            [0.9, 1.0, 0.8],
            [0.9, 0.8, 1.0],
        ], dtype=np.float32)
        selected = np.asarray([True, False, False])
        refined, stats = sparse_refine_labels(
            labels, det, expert, selected, 1, 1.0, 0.55, 0.01
        )
        self.assertEqual(refined.tolist(), labels.tolist())
        self.assertEqual(stats["moved_cases"], 0)

    def test_active_pairs_are_sparse_and_unique(self):
        selected = np.asarray([True, False, True, False, False])
        anchors = {0: [1], 1: [3, 4]}
        active = sorted({0, 1, 2, 3, 4})
        mapping = {value: idx for idx, value in enumerate(active)}
        local_pairs, global_pairs = build_active_pairs(selected, anchors, mapping)
        self.assertEqual(len(global_pairs), len(set(global_pairs)))
        self.assertEqual(len(local_pairs), 6)


if __name__ == "__main__":
    unittest.main()
