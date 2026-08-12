from __future__ import annotations

import argparse
import unittest

import numpy as np
import torch

import theta_trilog_model as ttm


class SupConTests(unittest.TestCase):
    def test_supcon_loss_finite_and_positive(self):
        features = torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.05, 0.95]], dtype=torch.float32
        )
        targets = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
        loss = ttm.supcon_loss(features, targets, temperature=0.1)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)

    def test_train_with_supcon_end_to_end(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(128, 40)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
        w = np.ones(128, dtype=np.float32)
        args = argparse.Namespace(
            random_state=0, device="cpu", epochs=2, batch_size=32, lr=1e-3,
            weight_decay=0.0, dropout=0.1, early_stop_patience=2,
            focal_gamma=2.0, fusion="concat", supcon_weight=0.5,
            supcon_temperature=0.1,
        )
        pkg = ttm.train_trilog_pair_model(X, y, w, base_dim=20, args=args)
        self.assertEqual(pkg["model_type"], "theta_trilog_mlp")
        probs = ttm.predict_trilog_pair_model(pkg, X)
        self.assertEqual(tuple(probs.shape), (128,))
        self.assertTrue(np.all((probs >= 0) & (probs <= 1)))


if __name__ == "__main__":
    unittest.main()
