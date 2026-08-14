from __future__ import annotations

import argparse
import unittest

import numpy as np

import theta_trilog_model as ttm


class FusionAblationTests(unittest.TestCase):
    def test_concat_and_sum_fusion_forward(self):
        import torch
        net_concat = ttm._build_network(400, 200, 0.1, fusion="concat")
        net_sum = ttm._build_network(400, 200, 0.1, fusion="sum")
        x = torch.randn(5, 400)
        self.assertEqual(tuple(net_concat(x).shape), (5,))
        self.assertEqual(tuple(net_sum(x).shape), (5,))
        self.assertEqual(net_concat.head[0].in_features, 256 * 4)
        self.assertEqual(net_sum.head[0].in_features, 256)
        # forward_features is the penultimate fused representation
        self.assertEqual(tuple(net_sum.forward_features(x).shape), (5, 256))

    def test_train_with_sum_fusion_end_to_end(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(64, 40)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
        w = np.ones(64, dtype=np.float32)
        args = argparse.Namespace(
            random_state=0, device="cpu", epochs=2, batch_size=16, lr=1e-3,
            weight_decay=0.0, dropout=0.1, early_stop_patience=2,
            focal_gamma=2.0, fusion="sum",
        )
        pkg = ttm.train_trilog_pair_model(X, y, w, base_dim=20, args=args)
        self.assertEqual(pkg["model_type"], "theta_trilog_mlp")
        self.assertEqual(pkg["fusion"], "sum")
        probs = ttm.predict_trilog_pair_model(pkg, X)
        self.assertEqual(tuple(probs.shape), (64,))
        self.assertTrue(np.all((probs >= 0) & (probs <= 1)))


if __name__ == "__main__":
    unittest.main()
