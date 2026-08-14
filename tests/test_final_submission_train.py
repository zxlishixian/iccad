from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_final_submission_train as rt

SMALL_DATASETS = [
    "dataset/fake_dataset/old_fake_dataset/first_batch_dataset",
    "dataset/fake_dataset/official_format_fake_dataset/directed_cross_v2",
    "dataset/fake_dataset/official_format_fake_dataset/stable_official_like_multitest_v1",
    "dataset/real_dataset/benchmark_set_1",
    "dataset/real_dataset/benchmark_set_2",
]


class FinalTrainSmokeTests(unittest.TestCase):
    def test_train_writes_models_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            # Unset the LLM config so the smoke test runs deterministic features.
            env = {k: v for k, v in os.environ.items() if k != "LLM_MODEL_CONFIG"}
            with mock.patch.dict(os.environ, env, clear=True):
                code = rt.main([
                    "--output-dir", str(out),
                    "--datasets", *SMALL_DATASETS,
                    "--seeds", "0",
                    "--pretrain-epochs", "2",
                    "--finetune-epochs", "2",
                    "--device", "cpu",
                    "--view-max-pairs-per-dataset", "200",
                ])
            self.assertEqual(code, 0)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(len(manifest["folds"]), 3)
            self.assertEqual(len(manifest["seeds"]), 1)
            for fold in manifest["folds"]:
                self.assertTrue((out / "models" / f"model_{fold}_seed0.pt").exists(), fold)
                self.assertTrue((out / "models" / f"preprocess_{fold}_seed0.pkl").exists(), fold)


if __name__ == "__main__":
    unittest.main()
