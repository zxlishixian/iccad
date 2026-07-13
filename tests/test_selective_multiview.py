from __future__ import annotations

import unittest

import numpy as np

import run_selective_multiview_experiments as sm


class SelectiveMultiviewTests(unittest.TestCase):
    def test_selection_obeys_fraction_and_cap(self):
        difficulty = np.linspace(0.0, 1.0, 100, dtype=np.float32)
        selected = sm.select_expert_cases(difficulty, 0.30, 2, 20)
        self.assertEqual(int(np.sum(selected)), 20)
        self.assertTrue(selected[-1])

    def test_both_endpoint_policy_changes_only_selected_block(self):
        dual = np.full((4, 4), 0.2, dtype=np.float32)
        five = np.full((4, 4), 0.8, dtype=np.float32)
        np.fill_diagonal(dual, 1.0)
        np.fill_diagonal(five, 1.0)
        selected = np.array([True, True, False, False])
        out, changed = sm.selective_probability(dual, five, selected, 1.0, "both")
        self.assertEqual(changed, 1)
        self.assertAlmostEqual(float(out[0, 1]), 0.8, places=6)
        self.assertAlmostEqual(float(out[0, 2]), 0.2, places=6)

    def test_neighbor_expansion_counts_new_cases_and_respects_cap(self):
        selected = np.array([True, False, False, False, False])
        similarity = np.array([
            [1.0, 0.9, 0.8, 0.2, 0.1],
            [0.9, 1.0, 0.7, 0.3, 0.2],
            [0.8, 0.7, 1.0, 0.4, 0.3],
            [0.2, 0.3, 0.4, 1.0, 0.8],
            [0.1, 0.2, 0.3, 0.8, 1.0],
        ], dtype=np.float32)
        expanded = sm.expand_with_neighbors(selected, similarity, 3, 3)
        self.assertEqual(int(np.sum(expanded)), 3)
        self.assertTrue(np.all(expanded[:3]))

    def test_difficulty_is_finite(self):
        base = np.array([
            [1.0, 0.8, 0.2, 0.1],
            [0.8, 1.0, 0.3, 0.2],
            [0.2, 0.3, 1.0, 0.7],
            [0.1, 0.2, 0.7, 1.0],
        ], dtype=np.float32)
        stack = np.stack([base, np.clip(base + 0.03, 0.0, 1.0)])
        score, components, mean_prob = sm.case_difficulty(stack, 2, "agglomerative_avg")
        self.assertEqual(score.shape, (4,))
        self.assertTrue(np.all(np.isfinite(score)))
        self.assertEqual(set(components), {"entropy", "disagreement", "instability", "margin_difficulty"})
        self.assertEqual(mean_prob.shape, (4, 4))


if __name__ == "__main__":
    unittest.main()
