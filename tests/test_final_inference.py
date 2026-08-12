from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import final_inference as fi
import official_style_features as osf
import run_final_submission_train as rt

SMALL_DATASETS = [
    "dataset/fake_dataset/old_fake_dataset/first_batch_dataset",
    "dataset/fake_dataset/official_format_fake_dataset/directed_cross_v2",
    "dataset/fake_dataset/official_format_fake_dataset/stable_official_like_multitest_v1",
    "dataset/real_dataset/benchmark_set_1",
    "dataset/real_dataset/benchmark_set_2",
]


class FinalInferenceSmokeTests(unittest.TestCase):
    def test_inference_writes_buckets(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            model_dir = tmp / "model"
            # Unset the LLM config so the smoke test runs deterministic features.
            env = {k: v for k, v in os.environ.items() if k != "LLM_MODEL_CONFIG"}
            with mock.patch.dict(os.environ, env, clear=True):
                code = rt.main([
                    "--output-dir", str(model_dir),
                    "--datasets", *SMALL_DATASETS,
                    "--seeds", "0",
                    "--pretrain-epochs", "2",
                    "--finetune-epochs", "2",
                    "--device", "cpu",
                    "--view-max-pairs-per-dataset", "200",
                ])
            self.assertEqual(code, 0)

            input_csv = Path("dataset/real_dataset/benchmark_set_1/input.csv")
            out_csv = tmp / "out.csv"
            fi.main([
                "--input", str(input_csv),
                "--output", str(out_csv),
                "--k", "2",
                "--model-dir", str(model_dir),
            ])
            with open(out_csv, newline="") as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0], ["Case", "bucket"])
            self.assertEqual(len(rows), 1 + len(osf.read_cases(input_csv)))
            self.assertTrue(all(row[1].startswith("bucket_") for row in rows[1:]))


if __name__ == "__main__":
    unittest.main()
