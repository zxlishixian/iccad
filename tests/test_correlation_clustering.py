from __future__ import annotations

import unittest

import numpy as np

import graph_clustering as gc


def _block(n, clusters, same=0.9, diff=0.1):
    prob = np.full((n, n), diff, dtype=np.float32)
    np.fill_diagonal(prob, 1.0)
    for cluster in clusters:
        for i in cluster:
            for j in cluster:
                if i != j:
                    prob[i, j] = prob[j, i] = same
    return prob


class CorrelationClusteringTests(unittest.TestCase):
    def test_recovers_three_clean_clusters(self):
        prob = _block(6, [[0, 1], [2, 3], [4, 5]])
        result = gc.correlation_cluster(prob)
        self.assertEqual(len(set(result.labels)), 3)
        self.assertEqual(result.labels[0], result.labels[1])
        self.assertNotEqual(result.labels[0], result.labels[2])
        self.assertEqual(result.labels[2], result.labels[3])
        self.assertEqual(result.labels[4], result.labels[5])
        self.assertNotEqual(result.labels[0], result.labels[4])

    def test_cannot_link_forces_split(self):
        prob = _block(2, [[0, 1]], same=0.95, diff=0.05)
        conflict = np.zeros((2, 2), dtype=np.float32)
        conflict[0, 1] = conflict[1, 0] = 1.0
        result = gc.correlation_cluster(prob, conflict_matrix=conflict)
        self.assertNotEqual(result.labels[0], result.labels[1])

    def test_soft_k_not_forced(self):
        prob = _block(6, [[0, 1], [2, 3], [4, 5]])
        result = gc.correlation_cluster(prob, k=2)
        self.assertEqual(len(set(result.labels)), 3)

    def test_single_case(self):
        result = gc.correlation_cluster(np.array([[1.0]], dtype=np.float32))
        self.assertEqual(result.labels, [0])

    def test_dispatch(self):
        prob = _block(6, [[0, 1], [2, 3], [4, 5]])
        result = gc.cluster_probability_graph(prob, 3, "correlation_cluster")
        self.assertEqual(result.method, "correlation_cluster")

    def test_fallback_on_degenerate(self):
        # All-similarity-0.5 matrix yields a degenerate result -> fallback used.
        prob = np.full((4, 4), 0.5, dtype=np.float32)
        np.fill_diagonal(prob, 1.0)
        result = gc.cluster_with_fallback(prob, 2)
        self.assertIn(result.method, ("correlation_cluster", "agglomerative_avg"))
        self.assertEqual(len(set(result.labels)), 2)


if __name__ == "__main__":
    unittest.main()
