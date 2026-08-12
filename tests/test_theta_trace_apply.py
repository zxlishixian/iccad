from __future__ import annotations

import unittest

import numpy as np

import theta_trace_features as ttf


class TraceApplyTests(unittest.TestCase):
    def test_dense_view_round_trip(self):
        rng = np.random.default_rng(0)
        matrix = rng.normal(size=(20, 40)).astype(np.float32)
        train = list(range(15))
        bundle, fitted = ttf._fit_dense_view(matrix, train, 8, seed=0)
        applied = ttf._apply_dense_view(matrix, bundle)
        self.assertEqual(applied.shape, (20, 8))
        np.testing.assert_allclose(applied, fitted, atol=1e-5)

    def test_text_view_round_trip(self):
        docs = ["alpha beta gamma delta"] * 10 + ["epsilon zeta eta"] * 10
        train = list(range(15))
        bundle, fitted = ttf._fit_text_view(docs, train, 8, seed=0)
        applied = ttf._apply_text_view(docs, bundle)
        self.assertEqual(applied.shape, (20, 8))
        np.testing.assert_allclose(applied, fitted, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
