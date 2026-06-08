from __future__ import annotations

import csv
import gzip
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import completion_case_features as ccf
import multigranular_features as mgf
import selective_expert as se
import signed_graph_clustering as sgc


class ExperimentalPipelineTests(unittest.TestCase):
    def make_dataset(self, root: Path, n: int = 2) -> Path:
        rows = []
        for i in range(n):
            case = root / f"case_{i+1}"
            case.mkdir(parents=True)
            (case / "regr.log").write_text(
                "start\nPC mismatch DUT retired : 80001234\nUVM_ERROR register write data mismatch to x5\n",
                encoding="utf-8",
            )
            with gzip.open(case / "sim.log.gz", "wt", encoding="utf-8") as f:
                f.write("boot\nUVM_FATAL @ 1200: timeout HANDLING_IRQ\nTEST FAILED\n")
            rows.append({"Case": str(i+1), "Regr Log": f"case_{i+1}/regr.log", "Sim Log": f"case_{i+1}/sim.log.gz", "Trace Log": ""})
        path = root / "input.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        return path

    def test_event_order_time_and_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_csv = self.make_dataset(Path(tmp), 1)
            evidence, _ = mgf.build_case_evidence([input_csv])
            self.assertEqual(len(evidence), 1)
            event_types = [x.event_type for x in evidence[0].events]
            self.assertIn("fatal", event_types)
            self.assertIn("pc_mismatch", event_types)
            self.assertIn("register_mismatch", event_types)
            fatal = next(x for x in evidence[0].events if x.event_type == "fatal")
            self.assertEqual(fatal.time, 1200)
            self.assertIn("0x80001", evidence[0].pc_regions)
            self.assertIn("x5", evidence[0].registers)

    def test_difficulty_budget(self):
        n = 40
        prob = np.full((n, n), 0.15, dtype=np.float32)
        np.fill_diagonal(prob, 1.0)
        labels = [i // 10 for i in range(n)]
        result = se.compute_case_difficulty(prob, labels, 4, budget=0.15, random_state=3)
        self.assertEqual(int(result.selected.sum()), 6)

    def test_completion_selection_and_429_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_csv = self.make_dataset(Path(tmp), 2)
            fake_module = types.ModuleType("openai")
            class Client:
                def __init__(self, **kwargs):
                    self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))
                def create(self, **kwargs):
                    raise RuntimeError("429 rate limit")
            fake_module.OpenAI = Client
            config = {"model":"x","base_url":"http://example","api_key":"x","timeout":1,"max_tokens":20}
            with mock.patch.object(ccf, "load_completion_config", return_value=config), mock.patch.dict(sys.modules, {"openai": fake_module}):
                features, _ = ccf.build_completion_case_features([input_csv], Path(tmp)/"cache", selected_indices={0})
            self.assertTrue(features[0].status.startswith("request_error"))
            self.assertEqual(features[1].status, "not_selected")

    def test_gate_missing_expert_strictly_falls_back(self):
        X = np.zeros((3, 14), dtype=np.float32)
        gate = se.predict_gate({"model_type":"constant","value":1.0}, X, np.zeros(3))
        self.assertTrue(np.all(gate == 0.0))
        base = np.asarray([[1,.2,.3],[.2,1,.4],[.3,.4,1]], dtype=np.float32)
        expert = np.ones((3,3), dtype=np.float32)
        pairs = [(0,1),(0,2),(1,2)]
        fused = se.fuse_probability_matrix(base, expert, pairs, gate)
        np.testing.assert_allclose(fused, base)

    def test_signed_graph_conflict_and_adaptive_bounds(self):
        prob = np.full((4,4), .1, dtype=np.float32)
        np.fill_diagonal(prob, 1.0)
        prob[0,1] = prob[1,0] = .99
        prob[2,3] = prob[3,2] = .80
        conflicts = np.zeros_like(prob)
        conflicts[0,1] = conflicts[1,0] = 1.0
        fixed = sgc.signed_graph_cluster(prob, 3, conflicts, k_policy="fixed", conflict_penalty=10.0)
        self.assertNotEqual(fixed.labels[0], fixed.labels[1])
        adaptive = sgc.signed_graph_cluster(prob, 3, conflicts, k_policy="adaptive")
        self.assertGreaterEqual(adaptive.selected_k, 2)
        self.assertLessEqual(adaptive.selected_k, 4)

    def test_gzip_official_columns_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_csv = self.make_dataset(root, 2)
            output = root / "pred.csv"
            subprocess.run([
                sys.executable, str(Path(__file__).resolve().parents[1]/"regr_fail_bucketing.py"),
                "--input", str(input_csv), "--output", str(output), "--k", "1",
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            with output.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(list(rows[0]), ["Case", "bucket"])
            self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
